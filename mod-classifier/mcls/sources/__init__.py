"""Dataset loaders. One module per dataset; each turns its files into `Recording`s.

Add a loader by subclassing `Source` and registering it in `REGISTRY`.
"""

from mcls.sources.base import REGISTRY, Source, get

__all__ = ["REGISTRY", "Source", "get"]
