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
"""Build and install the optional dispatcher and device libraries beside this file."""

import argparse
from pathlib import Path
import subprocess
import sysconfig

import torch
import torch_npu


def main() -> None:
    """Build against the active framework packages and sourced CANN installation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                    f"-DSOC_VERSION={args.soc}", f"-DTORCH_DIR={Path(torch.__file__).parent}",
                    f"-DTORCH_NPU_DIR={Path(torch_npu.__file__).parent}",
                    f"-DPYTHON_INCLUDE={sysconfig.get_path('include')}",
                    f"-DTORCH_ABI={int(torch.compiled_with_cxx11_abi())}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)
    subprocess.run(["cmake", "--install", str(build), "--prefix", str(source), "--component", "hp_runtime"], check=True)


if __name__ == "__main__":
    main()
