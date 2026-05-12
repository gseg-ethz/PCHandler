"""Regression test for COUPLE-01: SingletonMeta unification across worktrees."""
from GSEGUtils.singleton import SingletonMeta

from pchandler.geometry.optimal_shift import OptimizedShiftManager


def test_optimized_shift_manager_uses_shared_singleton_registry() -> None:
    """OptimizedShiftManager registers in the cross-repo SingletonMeta._instances dict.

    Validates that pchandler imports SingletonMeta from GSEGUtils (not from its own
    inline definition). After Phase 1 COUPLE-01, the two SingletonMeta classes are
    unified and OptimizedShiftManager appears in the shared registry.
    """
    instance = OptimizedShiftManager()
    assert OptimizedShiftManager in SingletonMeta._instances
    assert SingletonMeta._instances[OptimizedShiftManager] is instance
