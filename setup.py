# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""setup package."""
import sys
import logging
import os
import shutil
import stat
import platform
import subprocess
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from setuptools import setup, find_packages, find_namespace_packages, Distribution, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py
from setuptools.command.install import install

ROOT_DIR = os.path.dirname(__file__)
SYMM_GEMM_PACKAGE = "hyper_parallel.components.functional._ascendc.symmetric_matmul"
SYMM_GEMM_LIBRARIES = ("libhp_pipeline_kernel", "libhp_symm_torch")
logger = logging.getLogger(__name__)

TORCH26_REQUIRES = [
    "torch==2.6.0",
    "torch-npu==2.6.0.post3",
]

TORCH27_REQUIRES = [
    "torch==2.7.1",
    "torch-npu==2.7.1",
]

TORCH29_REQUIRES = [
    "torch==2.9.1",
    "torch-npu==2.9.1",
]

def _read_requirements(requirements_path: str) -> list[str]:
    """Read Python requirement lines from a repository-local file."""
    with open(os.path.join(ROOT_DIR, requirements_path), encoding='utf-8') as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]


def get_readme_content() -> str:
    """Read and return the contents of README.md for use as the package long description."""
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, 'README.md'), encoding='UTF-8') as f:
        return f.read()


def get_platform() -> str:
    """
    Get platform name.

    Returns:
        str, platform name in lowercase.
    """
    return f"{platform.system().strip().lower()}_{platform.machine().strip().lower()}"


def get_description() -> str:
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    os_info = get_platform()
    cpu_info = platform.machine().strip()

    return f'hyper_parallel platform: {os_info}, cpu: {cpu_info}'


def get_install_requires() -> list[str]:
    """
    Get install requirements.

    Returns:
        list, list of dependent packages.
    """
    return _read_requirements('requirements.txt')


def get_extra_requires() -> dict[str, list[str]]:
    """
    Get optional framework requirements.

    Returns:
        dict, optional dependency groups keyed by extra name.
    """
    return {
        "torch": list(TORCH29_REQUIRES),
        "torch26": list(TORCH26_REQUIRES),
        "torch27": list(TORCH27_REQUIRES),
        "torch29": list(TORCH29_REQUIRES),
        "all": list(TORCH29_REQUIRES),
    }


def update_permissions(path: str) -> None:
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IEXEC | stat.S_IWRITE)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD | stat.S_IWRITE)


def write_commit_id(target_lib_dir: str) -> None:
    """Write repository revision information into the wheel build tree.

    Args:
        target_lib_dir: Built ``hyper_parallel`` package directory.
    """
    commit_id_path = os.path.join(target_lib_dir, '.commit_id')
    try:
        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        commit = subprocess.run(
            ['git', 'log', '--abbrev-commit', '-1'],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        content = branch + commit
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        logger.warning('Cannot get Git revision information: %s', exc)
        content = 'git is not available while building.\n'
    with open(commit_id_path, 'w', encoding='utf-8') as commit_id_file:
        commit_id_file.write(content)


class EggInfo(egg_info):
    """Egg info."""

    def run(self) -> None:
        """Regenerate package metadata and normalize its permissions."""
        egg_info_dir = os.path.join(os.path.dirname(
            __file__), 'hyper_parallel.egg-info')
        shutil.rmtree(egg_info_dir, ignore_errors=True)
        super().run()
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """Build py files."""

    def run(self) -> None:
        """Build Python sources and copy the explicitly prepared indexed helper."""
        shutil.rmtree(self.build_lib, ignore_errors=True)
        super().run()
        target_lib_dir = os.path.join(self.build_lib, 'hyper_parallel')
        native_payload = os.environ.get("HYPER_PARALLEL_NATIVE_OUTPUT_ROOT", "").strip()
        if native_payload:
            if not os.path.isdir(native_payload):
                logger.warning(
                    "[HP-NATIVE-PAYLOAD-MISSING] wheel will contain only successfully built Python sources: %s",
                    native_payload,
                )
            else:
                shutil.copytree(
                    os.path.join(native_payload, "data", "indexed"),
                    os.path.join(target_lib_dir, "data", "indexed"),
                    dirs_exist_ok=True,
                )
                logger.info("Copied optional native payload from %s", native_payload)
        else:
            logger.info("No native payload selected; assembling a core-only wheel.")
        write_commit_id(target_lib_dir)
        update_permissions(target_lib_dir)


class BuildExt(build_ext):
    """Build the Ascend C libraries through the component's CMake entry point."""

    def get_ext_filename(self, fullname: str) -> str:
        """Keep the shared library names expected by the operator loader.

        Args:
            fullname: Qualified extension name or its final component.

        Returns:
            The extension filename, including package directories for qualified names.
        """
        parts = fullname.split(".")
        if parts[-1] in SYMM_GEMM_LIBRARIES:
            parts[-1] += ".so"
            return os.path.join(*parts)
        return super().get_ext_filename(fullname)

    def build_extensions(self) -> None:
        """Place both artifacts in the standard wheel or editable build directory."""
        source = Path(ROOT_DIR).joinpath(*SYMM_GEMM_PACKAGE.split("."))
        destination = Path(self.get_ext_fullpath(f"{SYMM_GEMM_PACKAGE}.libhp_symm_torch")).resolve().parent
        # Pip isolates its build backend through PYTHONPATH. CMake must instead
        # link against the Torch and torch-npu installed in the target interpreter.
        with TemporaryDirectory(prefix="hp-symm-build-") as build_dir:
            staging = Path(build_dir) / "runtime"
            subprocess.run([sys.executable, "-E", str(source / "build.py"), "--optional",
                            "--build-dir", build_dir, "--install-dir", str(staging)], check=True)
            if not staging.is_dir():
                self.extensions = []
                return
            destination.mkdir(parents=True, exist_ok=True)
            for name in SYMM_GEMM_LIBRARIES:
                shutil.copy2(staging / f"{name}.so", destination)


class Install(install):
    """Install."""

    def run(self) -> None:
        """Install the package and normalize installed-file permissions."""
        super().run()
        if sys.argv[-1] == 'install':
            pip = import_module('pip')
            hyper_parallel_dir = os.path.join(
                os.path.dirname(pip.__path__[0]), 'hyper_parallel')
            update_permissions(hyper_parallel_dir)


class BinaryDistribution(Distribution):
    """Force wheel to be tagged as a binary distribution.

    The package ships pre-compiled .so files prepared by component build entries
    and merged from their payloads. Without this hint setuptools/wheel would label
    the wheel as pure-python (py3-none-any), which lets pip install it under
    incompatible Python versions or CPU architectures and triggers
    'Python version mismatch' at import time.
    """

    def has_ext_modules(self) -> bool:
        """Return True so wheel is tagged for a specific cpython + platform."""
        return True


if __name__ == '__main__':
    _cmdclass = {
        'egg_info': EggInfo,
        'build_py': BuildPy,
        'build_ext': BuildExt,
        'install': Install,
    }
    setup(
        name='hyper_parallel',
        version='1.2.0.dev0',
        author='The MindSpore Authors',
        url='https://www.mindspore.cn',
        download_url='https://gitcode.com/mindspore/hyper-parallel/tags',
        project_urls={
            'Sources': 'https://gitcode.com/mindspore/hyper-parallel',
            'Issue Tracker': 'https://gitcode.com/mindspore/hyper-parallel/issues',
        },
        description=get_description(),
        long_description=get_readme_content(),
        long_description_content_type="text/markdown",
        test_suite="tests",
        packages=find_packages(exclude=["*tests*"]) + find_namespace_packages(
            include=["hyper_parallel.rl", "hyper_parallel.rl.rl", "hyper_parallel.rl.rl.*",
                     "hyper_parallel.rl.examples", "hyper_parallel.rl.examples.*",
                     "examples.torch.module_family", "examples.torch.module_family.*"],
        ),
        platforms=[get_platform()],
        include_package_data=True,
        package_data={
            'hyper_parallel.core.shard.ops': ['yaml/*.yaml'],
            'hyper_parallel.components.functional._triton.gated_delta_net': ['GDN_LICENSE'],
            'hyper_parallel.components.functional._triton.kimi_delta_attention': ['KDA_README.md'],
            # Model adapter recipes + committed codegen reference bundles.
            'hyper_parallel.models.qwen3_moe': [
                'recipes/*.yaml', 'generated/*.json', 'generated/*.diff',
            ],
            'hyper_parallel.models.deepseek_v41': [
                'generated/*.json', 'generated/*.diff',
            ],
            'hyper_parallel.models.deepseek_v41.tools': ['*.sh', '*.yaml', '*.md'],
            'hyper_parallel.auto_parallel': ['op_profiles/*.yaml', 'op_records.yaml'],
            'hyper_parallel.models.deepseek_v3': ['recipes/*.yaml'],
            'hyper_parallel.auto_parallel.sapp_nd.memory_estimation': [
                'configs_eval/default.yaml',
            ],
            'hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers': [
                'mapping.yaml',
            ],
        },
        ext_modules=[Extension(f"{SYMM_GEMM_PACKAGE}.{name}", sources=[]) for name in SYMM_GEMM_LIBRARIES]
        if os.environ.get("ASCEND_HOME_PATH") or os.environ.get("HYPER_PARALLEL_ASCEND_SOC") else [],
        cmdclass=_cmdclass,
        distclass=BinaryDistribution,
        python_requires='>=3.10,<3.13',
        install_requires=get_install_requires(),
        extras_require=get_extra_requires(),
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Web Environment',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords='hyper_parallel',
    )
