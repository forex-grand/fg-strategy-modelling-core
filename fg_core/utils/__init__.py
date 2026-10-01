"""Utility modules for performance testing and client interactions.

This package contains utility functions and classes for:
- Benchmark performance measurements (benchmark)
- Integration with ForexGrand tester service (fg_tester_client)
"""

from fg_core.utils.benchmark import benchmark
from fg_core.utils.fg_tester_client import FGTesterClient

__all__ = [
    "benchmark",
    "FGTesterClient",
]
