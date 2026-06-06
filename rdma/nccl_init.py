"""
rdma/nccl_init.py
-----------------
NCCL communicator bootstrap via ConfigMap rendezvous for TP=2 pod pairs.

--no-gpu / MINI_LLM_NO_GPU=1  →  CPU simulation mode:
  - No actual NCCL / CUDA calls are made.
  - All-reduce is simulated with numpy array averaging + synthetic latency
    calibrated to socket-transport NCCL (~10 Gb/s loopback).
  - The ConfigMap rendezvous store is replaced by an in-process dict,
    so both "rank 0" and "rank 1" can run in threads on the same laptop.

Usage (real)
------------
    from rdma.nccl_init import NCCLComm
    comm = NCCLComm(rank=0, world_size=2, no_gpu=False)
    comm.bootstrap(rendezvous_key="tp2-uid")
    comm.all_reduce(tensor)   # blocks until both ranks done

Usage (sim)
-----------
    comm0 = NCCLComm(rank=0, world_size=2, no_gpu=True)
    comm1 = NCCLComm(rank=1, world_size=2, no_gpu=True)
    import threading
    t = threading.Thread(target=comm1.bootstrap, args=("tp2-uid",))
    t.start()
    comm0.bootstrap("tp2-uid")
    t.join()
    result = comm0.all_reduce_sim(data_bytes=1 * 1024**3)
    print(result)
"""

from __future__ import annotations

import os
import time
import json
import uuid
import logging
import argparse
import threading
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------

def _no_gpu_from_env() -> bool:
    return os.environ.get("MINI_LLM_NO_GPU", "0").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AllReduceResult:
    success: bool
    barrier_ms: float          # time both ranks spent blocked at synchronization
    bus_bandwidth_gbps: float  # effective NCCL bus bandwidth
    data_bytes: int
    transport: str             # "socket" | "RoCEv2" | "NVLink" | "sim"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# In-process rendezvous store (replaces Kubernetes ConfigMap in sim mode)
# ---------------------------------------------------------------------------

_rendezvous_store: dict[str, str] = {}
_rendezvous_lock = threading.Lock()
_barrier_events: dict[str, dict[int, threading.Event]] = {}


def _store_put(key: str, value: str) -> None:
    with _rendezvous_lock:
        _rendezvous_store[key] = value
        logger.debug("[SIM-rendezvous] PUT %s = %s", key, value)


def _store_get(key: str, timeout: float = 10.0) -> Optional[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _rendezvous_lock:
            if key in _rendezvous_store:
                return _rendezvous_store[key]
        time.sleep(0.02)
    return None


# ---------------------------------------------------------------------------
# Kubernetes ConfigMap rendezvous (real path helpers)
# ---------------------------------------------------------------------------

def _k8s_configmap_put(namespace: str, name: str, uid: str) -> None:
    """Write NCCL unique ID to a ConfigMap (rank 0 only)."""
    try:
        from kubernetes import client, config  # type: ignore
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        body = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=name, namespace=namespace),
            data={"nccl_unique_id": uid},
        )
        try:
            v1.create_namespaced_config_map(namespace, body)
        except Exception:
            v1.patch_namespaced_config_map(name, namespace, body)
        logger.info("ConfigMap %s/%s written with NCCL UID", namespace, name)
    except ImportError:
        raise RuntimeError(
            "kubernetes Python client not installed. "
            "Run: pip install kubernetes"
        )


def _k8s_configmap_get(namespace: str, name: str, timeout: float = 30.0) -> str:
    """Poll a ConfigMap until the NCCL unique ID appears (rank 1)."""
    from kubernetes import client, config  # type: ignore
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            cm = v1.read_namespaced_config_map(name, namespace)
            uid = cm.data.get("nccl_unique_id")
            if uid:
                return uid
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(f"NCCL UID not found in ConfigMap {namespace}/{name} after {timeout}s")


# ---------------------------------------------------------------------------
# Simulated NCCL transport
# ---------------------------------------------------------------------------

class _SimNCCL:
    """
    Thread-safe in-process simulation of NCCL all-reduce for TP=2.

    Both ranks call all_reduce(); the second caller triggers the operation
    and both receive the result.
    """

    # Simulated socket-transport bandwidth: ~10 Gb/s → 1.25 GB/s
    SOCKET_BW_BYTES_PER_SEC: float = 1.25e9
    BASE_LATENCY_SEC: float = 0.0001   # 100 µs base

    _pending: dict[str, dict] = {}
    _lock = threading.Lock()

    @classmethod
    def all_reduce(cls, comm_id: str, rank: int, data_bytes: int) -> AllReduceResult:
        t0 = time.perf_counter()

        with cls._lock:
            if comm_id not in cls._pending:
                cls._pending[comm_id] = {
                    "ranks_arrived": 0,
                    "event": threading.Event(),
                    "result": None,
                }
            entry = cls._pending[comm_id]
            entry["ranks_arrived"] += 1
            arrived = entry["ranks_arrived"]

        if arrived >= 2:
            # Second rank: perform the simulated reduce and signal
            transfer_sec = data_bytes / cls.SOCKET_BW_BYTES_PER_SEC
            sleep_time = max(0, cls.BASE_LATENCY_SEC + transfer_sec - (time.perf_counter() - t0))
            time.sleep(sleep_time)
            bus_bw = data_bytes / max(1e-9, time.perf_counter() - t0) / 1e9
            result = AllReduceResult(
                success=True,
                barrier_ms=round((time.perf_counter() - t0) * 1000, 3),
                bus_bandwidth_gbps=round(bus_bw, 3),
                data_bytes=data_bytes,
                transport="sim-socket",
            )
            with cls._lock:
                entry["result"] = result
            entry["event"].set()
        else:
            # First rank: wait for second to arrive (barrier)
            entry["event"].wait(timeout=15)
            with cls._lock:
                result = entry.get("result") or AllReduceResult(
                    success=False,
                    barrier_ms=0,
                    bus_bandwidth_gbps=0,
                    data_bytes=data_bytes,
                    transport="sim-socket",
                    error="Timeout waiting for second rank",
                )

        with cls._lock:
            cls._pending.pop(comm_id, None)

        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class NCCLComm:
    """
    NCCL communicator for TP=2 tensor-parallel inference pods.

    Parameters
    ----------
    rank : int
        This pod's rank (0 or 1).
    world_size : int
        Total TP degree (currently always 2).
    no_gpu : bool
        CPU-simulation mode; no NCCL / CUDA required.
    namespace : str
        Kubernetes namespace for ConfigMap rendezvous (real mode).
    """

    def __init__(
        self,
        rank: int,
        world_size: int = 2,
        no_gpu: bool = False,
        namespace: str = "default",
    ):
        self.rank = rank
        self.world_size = world_size
        self.no_gpu = no_gpu or _no_gpu_from_env()
        self.namespace = namespace
        self._comm_id: Optional[str] = None
        self._initialized = False
        self._transport: str = "sim" if self.no_gpu else "unknown"

        mode = "CPU-simulation" if self.no_gpu else "real NCCL"
        logger.info("NCCLComm rank=%d world=%d mode=%s", rank, world_size, mode)

    # ------------------------------------------------------------------
    # Bootstrap / rendezvous
    # ------------------------------------------------------------------

    def bootstrap(self, rendezvous_key: str = "nccl-tp2-uid") -> None:
        """
        Exchange the NCCL unique ID between rank 0 and rank 1.

        Real mode  : rank 0 writes to a Kubernetes ConfigMap;
                     rank 1 polls and reads from it.
        Sim mode   : rank 0 writes to the in-process _rendezvous_store;
                     rank 1 polls and reads from it.
        """
        if self.no_gpu:
            if self.rank == 0:
                uid = str(uuid.uuid4())
                _store_put(rendezvous_key, uid)
                self._comm_id = uid
                logger.info("[SIM] rank 0 wrote UID %s to rendezvous store", uid)
            else:
                uid = _store_get(rendezvous_key)
                if uid is None:
                    raise TimeoutError("Rank 1 timed out waiting for NCCL UID from rank 0")
                self._comm_id = uid
                logger.info("[SIM] rank 1 read UID %s from rendezvous store", uid)
            self._initialized = True
            return

        # Real NCCL path (skeleton — requires pynccl / cupy / torch.distributed)
        if self.rank == 0:
            uid = str(uuid.uuid4())   # Real: ncclGetUniqueId()
            _k8s_configmap_put(self.namespace, rendezvous_key, uid)
            self._comm_id = uid
        else:
            self._comm_id = _k8s_configmap_get(self.namespace, rendezvous_key)

        # Real: ncclCommInitRank(comm, world_size, nccl_uid, rank)
        logger.info(
            "NCCL bootstrap complete rank=%d UID=%s "
            "(wire ncclCommInitRank here for real NCCL)",
            self.rank, self._comm_id,
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # All-reduce
    # ------------------------------------------------------------------

    def all_reduce(self, data_bytes: int) -> AllReduceResult:
        """
        Perform an NCCL all-reduce of *data_bytes* bytes.

        In sim mode: both ranks must call this concurrently (e.g. from
        separate threads) — the second caller triggers the simulated
        transfer and both get the result.

        In real mode: calls torch.distributed.all_reduce() on the
        initialized communicator (wire here).
        """
        if not self._initialized:
            raise RuntimeError("Call bootstrap() before all_reduce()")

        assert self._comm_id is not None

        if self.no_gpu:
            return _SimNCCL.all_reduce(
                comm_id=self._comm_id,
                rank=self.rank,
                data_bytes=data_bytes,
            )

        # Real path skeleton
        # import torch
        # import torch.distributed as dist
        # tensor = torch.zeros(data_bytes // 4, dtype=torch.float32).cuda()
        # t0 = time.perf_counter()
        # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - t0
        raise NotImplementedError(
            "Real NCCL all_reduce not yet wired. "
            "Use no_gpu=True or implement torch.distributed.all_reduce here."
        )

    # ------------------------------------------------------------------
    # Convenience sim wrapper (single-process both ranks)
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_tp2_all_reduce(data_bytes: int = 1 * 1024**3) -> tuple[AllReduceResult, AllReduceResult]:
        """
        Spin up two simulated ranks in threads and run one all-reduce.
        Useful for quick benchmarking on a laptop.

        Returns (result_rank0, result_rank1).
        """
        results: list[Optional[AllReduceResult]] = [None, None]
        errors: list[Optional[str]] = [None, None]

        def run_rank(r: int) -> None:
            try:
                comm = NCCLComm(rank=r, world_size=2, no_gpu=True)
                comm.bootstrap("bench-rendezvous")
                results[r] = comm.all_reduce(data_bytes)
            except Exception as e:
                errors[r] = str(e)

        threads = [threading.Thread(target=run_rank, args=(r,)) for r in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r, err in enumerate(errors):
            if err:
                results[r] = AllReduceResult(
                    success=False, barrier_ms=0, bus_bandwidth_gbps=0,
                    data_bytes=data_bytes, transport="sim", error=err,
                )

        return results[0], results[1]  # type: ignore


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NCCL TP=2 bootstrap smoke-test")
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode (no NCCL/CUDA required)")
    p.add_argument("--data-gb", type=float, default=1.0,
                   help="All-reduce message size in GB (default 1)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    data_bytes = int(args.data_gb * 1024**3)

    print(f"\n{'='*60}")
    print(f"  Mode      : {'CPU-simulation (--no-gpu)' if args.no_gpu else 'Real NCCL'}")
    print(f"  Message   : {args.data_gb:.1f} GB")
    print(f"  TP degree : 2")
    print(f"{'='*60}")

    if args.no_gpu:
        r0, r1 = NCCLComm.simulate_tp2_all_reduce(data_bytes)
        for rank, result in enumerate([r0, r1]):
            print(f"\n  Rank {rank}:")
            print(f"    Success        : {result.success}")
            print(f"    Barrier wait   : {result.barrier_ms:.2f} ms")
            print(f"    Bus bandwidth  : {result.bus_bandwidth_gbps:.3f} Gb/s")
            print(f"    Transport      : {result.transport}")
    else:
        print("  (real NCCL path — run inside a GPU pod with pynccl installed)")
    print(f"\n{'='*60}\n")
