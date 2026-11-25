# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Build the host-only Muon extension and bind Triton-Ascend's generated launcher ABI."""

import ctypes
import hashlib
import importlib.util
import os
import shlex
import subprocess
import sysconfig
import tempfile
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import torch
import torch_npu
from filelock import FileLock
from triton.backends.ascend import driver
from triton.backends.ascend.utils import get_backend_func


class UnsupportedLauncher(Exception):
    """The installed backend does not generate the paired-launch ABI."""


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_host_extension() -> ModuleType:
    """Build once per source, Python, torch, torch-npu and CANN installation."""
    if version("triton-ascend").split(".")[:2] != ["3", "6"] or not all(
        hasattr(driver, name) for name in ("make_launcher", "make_npu_launcher_stub", "generate_npu_header_src")
    ):
        raise UnsupportedLauncher("Requires the Triton-Ascend 3.6 launcher ABI")
    source = Path(__file__).with_name("muon_host.cpp")
    torch_path = Path(torch.__file__).resolve().parent
    npu_path = Path(torch_npu.__file__).resolve().parent
    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann:
        raise UnsupportedLauncher("ASCEND_HOME_PATH is unset")
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    abi = int(torch.compiled_with_cxx11_abi())
    environment = (torch.__version__, torch_npu.__version__, torch_path, npu_path, cann, compiler, abi, suffix)
    digest = hashlib.sha256(source.read_bytes() + repr(environment).encode()).hexdigest()
    cache = Path(os.getenv("TRITON_CACHE_DIR") or Path.home() / ".triton" / "cache") / digest
    cache.mkdir(parents=True, exist_ok=True)
    name = "_hp_muon_host"
    library = cache / (name + suffix)
    with FileLock(str(cache / "build.lock")):
        if not library.is_file():
            includes = [torch_path / "include", torch_path / "include/torch/csrc/api/include",
                        npu_path / "include", Path(cann) / "include", Path(cann) / "include/experiment",
                        Path(sysconfig.get_path("include"))]
            with tempfile.TemporaryDirectory(dir=cache) as build:
                output = Path(build) / library.name
                subprocess.run([*compiler, "-O2", "-std=c++20", "-shared", "-fPIC", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
                                *(f"-I{path}" for path in includes), str(source), "-o", str(output),
                                f"-L{torch_path / 'lib'}", f"-L{npu_path / 'lib'}", "-ltorch_python", "-ltorch_npu",
                                "-ltorch_cpu", "-lc10"], check=True)
                output.replace(library)
    return _load_module(name, library)


_SIGNATURE = """void triton_launch_kernel(const char* kernelName, cann_func_handle func, cann_stream stream,
    int gridX, int gridY, int gridZ, const int64_t* shapes_data, const int* shape_dims, int num_tensors,
    const int* tensor_kinds, const void* const* kernel_args, const size_t* arg_sizes, int num_args)"""


def _replace_once(source: str, original: str, replacement: str) -> str:
    if source.count(original) != 1:
        raise UnsupportedLauncher("Unrecognized Triton-Ascend launch template")
    return source.replace(original, replacement, 1)


def _paired_wrapper(wrapper: str, workspace: int) -> str:
    """Keep device-specific packing, lock setup and profiler callbacks from the installed driver."""
    start = wrapper.find("void triton_launch_kernel(")
    end = wrapper.find('} // extern "C"', start)
    body = wrapper.find("{", start, end)
    if min(start, end, body) < 0 or wrapper[start:body].split() != _SIGNATURE.split():
        raise UnsupportedLauncher("Unrecognized Triton-Ascend launcher signature")
    entry = wrapper[start:end]
    entry = _replace_once(entry, "void triton_launch_kernel(", "void hp_muon_launch(")
    entry = _replace_once(entry, "int num_args) {", "int num_args, void* scratch, uint64_t capacity, int* status) {")
    if workspace:
        entry = _replace_once(entry, get_backend_func("allocate_memory", "totalWorkSpaceSize", "stream"),
                              "if (capacity < totalWorkSpaceSize) return;\n  workspace_addr_ptr = scratch;")
        entry = _replace_once(entry, "std::shared_ptr<void> workspace_handle_guard(workspace_handle, "
                                    "release_npu_tensor_handle);", "")
    entry = _replace_once(entry, get_backend_func("async_launch", "launch_call"), "*status = static_cast<int>(launch_call());")
    return wrapper[:start] + entry + wrapper[end:]


def bind_product(extension: ModuleType, compiled, cores: int, lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor):
    """Return the callable plan and the owners of its kernel and launcher handles."""
    source = compiled.src

    def index(key):
        return source.fn.arg_names.index(key) if isinstance(key, str) else key

    signature = {index(key): value for key, value in source.signature.items()}
    if [value for value in signature.values() if value != "constexpr"] != ["*bf16"] * 3:
        raise UnsupportedLauncher("Expected three BF16 pointers")
    constants = {index(key): value for key, value in source.constants.items()}
    metadata = compiled.metadata
    workspace = getattr(metadata, "workspace_size", 0)
    wrapper = _paired_wrapper(driver.make_launcher(constants, signature, metadata), workspace)
    path = driver.make_npu_launcher_stub(driver.generate_npu_header_src(), wrapper, metadata.debug)
    module = _load_module("__triton_launcher", Path(path))
    library = ctypes.CDLL(path)
    # Triton initializes the function handle and packed metadata on the first access to its runner.
    runner = compiled.run
    packed = compiled.packed_metadata
    plan = extension.ProductPlan(ctypes.cast(library.hp_muon_launch, ctypes.c_void_p).value, compiled.function,
                                 packed["kernel_name"], cores, workspace * cores, packed["tensor_kinds"], lhs, rhs, out)
    return plan, compiled, runner, module, library
