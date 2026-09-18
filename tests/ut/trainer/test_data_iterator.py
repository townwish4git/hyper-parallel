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
"""Unit tests for trainer-side dataloader iteration helpers."""

import threading
import unittest

from hyper_parallel.trainer.runtime.data_iterator import BackgroundPrefetcher, HyperIter
from tests.common.mark_utils import arg_mark


class _BlockingDataLoader:
    """Block one iterator advance until a test releases it."""

    def __init__(self) -> None:
        """Initialize worker coordination events."""
        self.next_started = threading.Event()
        self.release_next = threading.Event()

    def __iter__(self) -> "_BlockingDataLoader":
        """Return this dataloader as its own iterator."""
        return self

    def __next__(self) -> object:
        """Wait for the test to release one iterator advance."""
        self.next_started.set()
        self.release_next.wait()
        return object()


class _TwoItemIterator:
    """Produce two items so the second queue write waits behind the first."""

    def __init__(self) -> None:
        """Initialize the item index and second-item notification."""
        self.index = 0
        self.second_item_ready = threading.Event()

    def __iter__(self) -> "_TwoItemIterator":
        """Return this object as its own iterator."""
        return self

    def __next__(self) -> int:
        """Return two items and signal before the second queue write."""
        self.index += 1
        if self.index == 2:
            self.second_item_ready.set()
        if self.index <= 2:
            return self.index
        raise StopIteration


class TestBackgroundPrefetcher(unittest.TestCase):
    """Tests for cooperative worker shutdown and reference release."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_timed_out_stop_reports_failure_then_worker_releases_references(self) -> None:
        """
        Feature: Background prefetch worker lifecycle.
        Description: Let a blocked worker outlive a finite stop timeout, then release it.
        Expectation: Stop reports the timeout and the worker eventually releases every live data reference.
        """
        dataloader = _BlockingDataLoader()
        prefetcher = BackgroundPrefetcher(dataloader)
        self.addCleanup(prefetcher.stop, 1.0)
        self.addCleanup(dataloader.release_next.set)
        self.assertTrue(dataloader.next_started.wait(timeout=1.0))

        with self.assertLogs("hyper_parallel.trainer.runtime.data_iterator", level="WARNING"):
            stopped = prefetcher.stop(timeout=0.01)

        self.assertFalse(stopped)
        self.assertTrue(prefetcher.thread.is_alive())

        dataloader.release_next.set()
        prefetcher.thread.join(timeout=1.0)

        self.assertFalse(prefetcher.thread.is_alive())
        self.assertTrue(prefetcher.queue.empty())
        self.assertIsNone(prefetcher.iterator)
        self.assertIsNone(prefetcher.dataloader)
        self.assertIsNone(prefetcher.original_state_dict)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_stop_unblocks_worker_waiting_to_write_to_full_queue(self) -> None:
        """
        Feature: Background prefetch worker lifecycle.
        Description: Stop a producer whose second item is waiting on a full queue.
        Expectation: Reliable stop drains the queue and waits for the worker to exit.
        """
        iterator = _TwoItemIterator()
        prefetcher = BackgroundPrefetcher(iterator, maxsize=1)
        self.addCleanup(prefetcher.stop, 1.0)
        self.assertTrue(iterator.second_item_ready.wait(timeout=1.0))

        self.assertTrue(prefetcher.stop(timeout=None))

        self.assertFalse(prefetcher.thread.is_alive())
        self.assertTrue(prefetcher.queue.empty())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_hyper_iter_stop_is_successful_without_background_prefetching(self) -> None:
        """
        Feature: Unified iterator stop result.
        Description: Stop an iterator configured without background prefetching.
        Expectation: Stop is an immediate successful no-op.
        """
        data_iterator = HyperIter([1, 2], use_background_prefetcher=False)

        self.assertTrue(data_iterator.stop())


if __name__ == "__main__":
    unittest.main()
