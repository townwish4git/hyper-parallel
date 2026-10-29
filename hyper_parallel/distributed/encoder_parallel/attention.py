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
"""Encoder attention segments and exact execution auditing."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment


@dataclass
class EncoderAttentionContext:
    """One ViT pass: segment boundaries, Ulysses mesh, and call audit."""

    segments: tuple[EncoderSegment, ...]
    cp_mesh: DeviceMesh
    declared_heads: Mapping[str, int]
    expected: frozenset[str]
    seen: dict[str, int] = field(default_factory=dict)

    def record(self, module_fqn: str) -> None:
        """Record one replacement invocation.

        Args:
            module_fqn: Final module path for the attention block.
        """
        self.seen[module_fqn] = self.seen.get(module_fqn, 0) + 1

    def assert_exactly_once(self) -> None:
        """Reject missing or repeated calls in an active ViT pass."""
        if set(self.seen) != self.expected or any(count != 1 for count in self.seen.values()):
            raise RuntimeError(
                "vision attention execution mismatch; "
                f"expected={sorted(self.expected)}; seen={self.seen}"
            )

    def validate_runtime_head_counts(self, *, module_fqn: str, query_heads: int,
                                     key_heads: int, value_heads: int) -> None:
        """Check projected head counts before entering a collective."""
        actual = {"query": query_heads, "key": key_heads, "value": value_heads}
        if actual != self.declared_heads or any(count % self.cp_mesh.size() for count in actual.values()):
            raise ValueError(
                "Ulysses runtime head count mismatch; phase=runtime; "
                f"module_fqn={module_fqn}; declared_head_counts={self.declared_heads}; "
                f"runtime_head_counts={actual}; encoder_cp_size={self.cp_mesh.size()}"
            )


_ACTIVE_CONTEXT: ContextVar[EncoderAttentionContext | None] = ContextVar("encoder_attention_context", default=None)


def current_encoder_attention_context() -> EncoderAttentionContext | None:
    """Return the complete attention state for the current ViT pass."""
    return _ACTIVE_CONTEXT.get()


@contextmanager
def activate_encoder_attention(context: EncoderAttentionContext) -> Iterator[EncoderAttentionContext]:
    """Scope attention work and its exact-call audit to one packed ViT pass.

    Args:
        context: Segments, executor, and expected module paths for this pass.
    """
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)


class _EncoderReplayContext:
    """Restore one block's execution geometry with a fresh audit on each replay."""

    def __init__(self, context: EncoderAttentionContext, expected: frozenset[str]) -> None:
        """Capture immutable geometry and the attention names of one block."""
        self.segments = context.segments
        self.cp_mesh = context.cp_mesh
        self.declared_heads = dict(context.declared_heads)
        self.expected = expected
        self.scope = None
        self.context = None

    def __enter__(self) -> EncoderAttentionContext:
        """Activate fresh replay state without changing the original forward audit."""
        self.context = EncoderAttentionContext(self.segments, self.cp_mesh, self.declared_heads, self.expected)
        self.scope = activate_encoder_attention(self.context)
        return self.scope.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        """Check the replay audit and always restore the enclosing context."""
        try:
            if exc_type is None:
                self.context.assert_exactly_once()
        finally:
            self.scope.__exit__(exc_type, exc, traceback)
            self.scope = None
            self.context = None
        return False


def encoder_checkpoint_contexts(module_fqn: str) -> tuple[Any, Any]:
    """Capture this invocation's CP geometry without retaining a batch or mutable audit.

    Args:
        module_fqn: Logical path of the checkpointed vision block.

    Returns:
        Original-forward and backward-replay contexts for non-reentrant checkpointing.
    """
    context = current_encoder_attention_context()
    if context is None:
        return nullcontext(), nullcontext()
    expected = frozenset(name for name in context.expected if name.startswith(f"{module_fqn}."))
    if not expected:
        raise ValueError(f"checkpoint region has no encoder attention: {module_fqn}")
    return nullcontext(), _EncoderReplayContext(context, expected)
