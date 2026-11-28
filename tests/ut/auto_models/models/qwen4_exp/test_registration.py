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
"""Registration smoke test for the generated model adapter."""

import unittest

from hyper_parallel.models.registry import get_model_adapter



class TestRegistration(unittest.TestCase):
    """Verify model-family lookup after registration import."""

    def test_registration(self) -> None:
        """Require the generated adapter to be discoverable."""
        self.assertEqual(get_model_adapter("qwen4_exp").architecture, "Qwen4ExpForConditionalGeneration")
