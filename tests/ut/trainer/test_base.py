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
"""Unit tests for trainer lifecycle cleanup."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.common.mark_utils import arg_mark


_BATCHING_MODULE = ModuleType("hyper_parallel.data.batching")
_BATCHING_MODULE.build_dataloader = MagicMock()

# This test covers trainer lifecycle handling, not optional dataloader construction.
with patch.dict(sys.modules, {"hyper_parallel.data.batching": _BATCHING_MODULE}):
    base_module = importlib.import_module("hyper_parallel.trainer.base")


class TestBaseTrainerCleanup(unittest.TestCase):
    """Tests for cleaning up data iterators across trainer exits."""

    @staticmethod
    def _make_trainer(train_epochs: int = 1) -> MagicMock:
        """Build the minimal trainer surface required by ``BaseTrainer.train``."""
        trainer = MagicMock()
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(
                use_background_prefetcher=True,
                drop_last=False,
            )
        )
        trainer.train_dataloader = MagicMock()
        trainer.local_rank = 0
        trainer.state = SimpleNamespace(global_step=0, epoch=0)
        trainer.train_iters = train_epochs
        trainer.train_epochs = train_epochs
        trainer.train_steps = 1
        return trainer

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_train_waits_for_old_iterator_before_starting_next_epoch(self) -> None:
        """
        Feature: BaseTrainer epoch lifecycle.
        Description: Run two epochs with separate background data iterators.
        Expectation: The first iterator stops reliably before the second one is created.
        """
        trainer = self._make_trainer(train_epochs=2)
        events = []
        data_iterators = [MagicMock(), MagicMock()]
        iterator_index = 0

        def _make_data_iterator(*_args: object, **_kwargs: object) -> MagicMock:
            """Return the next mocked data iterator and record its creation."""
            nonlocal iterator_index
            events.append("create")
            data_iterator = data_iterators[iterator_index]
            iterator_index += 1
            return data_iterator

        data_iterators[0].stop.side_effect = lambda **_kwargs: events.append("stop") or True

        with patch.object(base_module, "HyperIter", side_effect=_make_data_iterator):
            with patch.object(base_module, "print_device_mem_info"):
                with patch.object(base_module, "synchronize"):
                    base_module.BaseTrainer.train(trainer)

        self.assertEqual(events[:3], ["create", "stop", "create"])
        data_iterators[0].stop.assert_called_once_with(timeout=None)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_train_stops_data_iterator_when_training_or_callback_raises(self) -> None:
        """
        Feature: BaseTrainer exceptional lifecycle.
        Description: Raise from the training step and epoch-end callback while prefetching is enabled.
        Expectation: The active iterator stops reliably and each original exception propagates unchanged.
        """
        for failure_hook in ("train_step", "on_epoch_end"):
            with self.subTest(failure_hook=failure_hook):
                trainer = self._make_trainer()
                data_iterator = MagicMock()
                expected_error = RuntimeError(f"{failure_hook} failed")
                getattr(trainer, failure_hook).side_effect = expected_error

                with patch.object(base_module, "HyperIter", return_value=data_iterator):
                    with self.assertRaises(RuntimeError) as error_context:
                        base_module.BaseTrainer.train(trainer)

                self.assertIs(error_context.exception, expected_error)
                data_iterator.stop.assert_called_once_with(timeout=None)


if __name__ == "__main__":
    unittest.main()
