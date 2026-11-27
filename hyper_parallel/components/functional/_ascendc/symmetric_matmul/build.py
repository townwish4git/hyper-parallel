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
"""Build the optional Torch dispatcher and Ascend C device library."""

import argparse
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sysconfig


def main() -> None:
    """Build against the active framework packages and sourced CANN installation."""
    source = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc", default=os.environ.get("HYPER_PARALLEL_ASCEND_SOC"),
                        help="CANN SoC target. Defaults to the first visible NPU.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, default=source)
    parser.add_argument("--optional", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("ASCEND_HOME_PATH"):
        parser.error("Source the target CANN set_env.sh before building the Ascend C symmetric GEMM.")
    if find_spec("torch") is None or find_spec("torch_npu") is None:
        if args.optional and not args.soc:
            return
        parser.error("Install the target torch and torch_npu packages before building the Ascend C symmetric GEMM.")

    # Metadata and CPU-only builds can run before the optional framework is installed.
    import torch
    import torch_npu

    if not args.soc and not torch_npu.npu.is_available():
        if args.optional:
            return
        parser.error("Set HYPER_PARALLEL_ASCEND_SOC or pass --soc when no NPU is visible.")
    soc = args.soc or torch_npu.npu.get_device_name(0)
    build = args.build_dir.resolve()
    subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                    f"-DSOC_VERSION={soc}", f"-DTORCH_DIR={Path(torch.__file__).parent}",
                    f"-DTORCH_NPU_DIR={Path(torch_npu.__file__).parent}",
                    f"-DPYTHON_INCLUDE={sysconfig.get_path('include')}",
                    f"-DTORCH_ABI={int(torch.compiled_with_cxx11_abi())}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    subprocess.run(["cmake", "--install", str(build), "--prefix", str(args.install_dir.resolve()),
                    "--component", "hp_runtime"], check=True)


if __name__ == "__main__":
    main()
