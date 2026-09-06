# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from unittest.mock import patch

import pytest

from trl.import_utils import is_vllm_available


@pytest.mark.parametrize("version", ["0.25.0", "0.25.1", "0.25.1+cu129"])
def test_vllm_025_is_supported(version):
    with patch("trl.import_utils._is_package_available", return_value=(True, version)):
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            assert is_vllm_available()

    assert not caught_warnings


def test_vllm_version_above_validated_ceiling_warns():
    with patch("trl.import_utils._is_package_available", return_value=(True, "0.27.2")):
        with pytest.warns(UserWarning, match="0.19.0 to 0.27.1"):
            assert is_vllm_available()
