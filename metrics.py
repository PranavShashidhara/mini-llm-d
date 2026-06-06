"""
mini-llm-d router — per-pod metrics state.
Tracks queue depth, tokens/sec, and circuit-breaker state for each pod.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class PodMetrics:
    """Live metrics snapshot for a single inference pod."""

    endpoint: str
    queue_depth: int = 0
    active_requests: int = 0
    tokens_per_sec: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    circuit_open: bool = False        # True == pod is circuit-broken
    circuit_open_until: float = 0.0  # monotonic timestamp to re-close circuit

    # Cumulative counters for Prometheus export
    total_requests_routed: int = 0
    total_errors: int = 0

    def is_healthy(self) -> bool:
        now = time.monotonic()
        if self.circuit_open:
            if now >= self.circuit_open_until:
                self.circuit_open = False   # half-open: let next probe through
            else:
                return False
        return True

    def record_timeout(self, cooldown_secs: float = 30.0) -> None:
        self.circuit_open = True
        self.circuit_open_until = time.monotonic() + cooldown_secs
        self.total_errors += 1


class MetricsRegistry:
    """
    Thread-safe registry of all known pod endpoints and their live metrics.
    Updated by the background poller in router.py.
    """

    def __init__(self) -> None:
        self._pods: dict[str, PodMetrics] = {}
        self._lock = asyncio.Lock()

    async def register(self, endpoint: str) -> None:
        async with self._lock:
            if endpoint not in self._pods:
                self._pods[endpoint] = PodMetrics(endpoint=endpoint)

    async def update(self, endpoint: str, data: dict) -> None:
        async with self._lock:
            if endpoint not in self._pods:
                self._pods[endpoint] = PodMetrics(endpoint=endpoint)
            m = self._pods[endpoint]
            m.queue_depth = data.get("queue_depth", 0)
            m.active_requests = data.get("active_requests", 0)
            m.tokens_per_sec = data.get("tokens_per_sec", 0.0)
            m.last_seen = time.monotonic()
            m.circuit_open = False  # successful poll clears circuit

    async def record_timeout(self, endpoint: str) -> None:
        async with self._lock:
            if endpoint in self._pods:
                self._pods[endpoint].record_timeout()

    async def record_routed(self, endpoint: str) -> None:
        async with self._lock:
            if endpoint in self._pods:
                self._pods[endpoint].total_requests_routed += 1

    async def snapshot(self) -> list[PodMetrics]:
        """Return a shallow copy of all pod metrics (safe to read outside lock)."""
        async with self._lock:
            return list(self._pods.values())

    async def healthy_pods(self) -> list[PodMetrics]:
        async with self._lock:
            return [m for m in self._pods.values() if m.is_healthy()]

    async def prometheus_text(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []
        snapshot = await self.snapshot()

        # Aggregate metrics
        total_tps = sum(m.tokens_per_sec for m in snapshot if m.is_healthy())
        total_queue = sum(m.queue_depth for m in snapshot if m.is_healthy())
        healthy_count = sum(1 for m in snapshot if m.is_healthy())

        lines += [
            "# HELP tokens_per_second_total Aggregate tokens/sec across all healthy pods",
            "# TYPE tokens_per_second_total gauge",
            f"tokens_per_second_total {total_tps:.2f}",
            "",
            "# HELP tokens_per_second_per_pod Average tokens/sec per healthy pod (HPA target)",
            "# TYPE tokens_per_second_per_pod gauge",
            f"tokens_per_second_per_pod {(total_tps / healthy_count) if healthy_count else 0:.2f}",
            "",
            "# HELP router_queue_depth_total Aggregate queue depth across healthy pods",
            "# TYPE router_queue_depth_total gauge",
            f"router_queue_depth_total {total_queue}",
            "",
            "# HELP router_healthy_pods Number of healthy inference pods",
            "# TYPE router_healthy_pods gauge",
            f"router_healthy_pods {healthy_count}",
            "",
        ]

        # Per-pod metrics
        lines += [
            "# HELP pod_tokens_per_sec Tokens/sec on each inference pod",
            "# TYPE pod_tokens_per_sec gauge",
        ]
        for m in snapshot:
            label = f'endpoint="{m.endpoint}"'
            lines.append(f"pod_tokens_per_sec{{{label}}} {m.tokens_per_sec:.2f}")

        lines += [
            "",
            "# HELP pod_queue_depth Queue depth on each inference pod",
            "# TYPE pod_queue_depth gauge",
        ]
        for m in snapshot:
            label = f'endpoint="{m.endpoint}"'
            lines.append(f"pod_queue_depth{{{label}}} {m.queue_depth}")

        lines += [
            "",
            "# HELP pod_circuit_open Circuit-breaker state per pod (1=open/broken)",
            "# TYPE pod_circuit_open gauge",
        ]
        for m in snapshot:
            label = f'endpoint="{m.endpoint}"'
            lines.append(f"pod_circuit_open{{{label}}} {int(m.circuit_open)}")

        lines += [
            "",
            "# HELP pod_requests_routed_total Total requests routed to each pod",
            "# TYPE pod_requests_routed_total counter",
        ]
        for m in snapshot:
            label = f'endpoint="{m.endpoint}"'
            lines.append(f"pod_requests_routed_total{{{label}}} {m.total_requests_routed}")

        return "\n".join(lines) + "\n"
