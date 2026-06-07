"""
router/prefix_cache.py

Stateful prefix-hash → pod-endpoint mapping.

The router records which pod last processed each prefix hash so that
subsequent requests with the same prefix can be routed to a pod that
already has the relevant KV cache entries warm.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CacheEntry:
    pod_endpoint: str
    recorded_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class PrefixCache:
    """
    Thread-safe (asyncio-compatible) mapping of prefix_hash → pod_endpoint.

    Usage:
        cache = PrefixCache(max_age_secs=300)
        cache.record("abc123", "http://pod-0:8000")
        endpoint = cache.lookup("abc123")   # -> "http://pod-0:8000" or None
    """

    def __init__(self, max_age_secs: float = 300.0):
        self._map: dict[str, CacheEntry] = {}
        self._max_age = max_age_secs
        self._lock = asyncio.Lock()

    async def record(self, prefix_hash: str, pod_endpoint: str) -> None:
        """Record or update a prefix → pod mapping after routing a request."""
        async with self._lock:
            existing = self._map.get(prefix_hash)
            if existing:
                existing.pod_endpoint = pod_endpoint
                existing.recorded_at = time.monotonic()
            else:
                self._map[prefix_hash] = CacheEntry(pod_endpoint=pod_endpoint)

    async def lookup(self, prefix_hash: str) -> Optional[str]:
        """
        Return the pod endpoint for this prefix hash if a fresh entry exists,
        otherwise None.
        """
        async with self._lock:
            entry = self._map.get(prefix_hash)
            if entry is None:
                return None
            age = time.monotonic() - entry.recorded_at
            if age > self._max_age:
                del self._map[prefix_hash]
                return None
            entry.hit_count += 1
            return entry.pod_endpoint

    async def evict_stale(self, max_age_secs: Optional[float] = None) -> int:
        """
        Remove entries older than max_age_secs (defaults to self._max_age).
        Also removes entries whose pod is no longer in healthy_pods.
        Returns the number of evicted entries.
        """
        cutoff = max_age_secs if max_age_secs is not None else self._max_age
        now = time.monotonic()
        evicted = 0
        async with self._lock:
            stale = [k for k, v in self._map.items() if now - v.recorded_at > cutoff]
            for k in stale:
                del self._map[k]
                evicted += 1
        return evicted

    async def evict_pod(self, pod_endpoint: str) -> int:
        """Remove all entries pointing to a specific pod (e.g. on pod failure)."""
        evicted = 0
        async with self._lock:
            dead = [k for k, v in self._map.items() if v.pod_endpoint == pod_endpoint]
            for k in dead:
                del self._map[k]
                evicted += 1
        return evicted

    async def stats(self) -> dict:
        async with self._lock:
            total = len(self._map)
            hits = sum(e.hit_count for e in self._map.values())
            pods: dict[str, int] = {}
            for e in self._map.values():
                pods[e.pod_endpoint] = pods.get(e.pod_endpoint, 0) + 1
        return {
            "total_entries": total,
            "total_hits": hits,
            "entries_per_pod": pods,
        }
