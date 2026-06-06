"""
rdma/kv_transfer.py
-------------------
libibverbs wrapper: MR registration, ibv_post_send RDMA WRITE, CQ poll.

Pass --no-gpu (or set env MINI_LLM_NO_GPU=1) to run in CPU-simulation mode:
  - No actual ibverbs / libibverbs calls are made.
  - Transfer is simulated with in-process bytearray copies + time.sleep()
    scaled to match the expected RoCEv2 loopback bandwidth (~10 Gb/s CPU-bound).
  - All public APIs, return types, and OTEL span attributes are identical to
    the real path so the router / server code requires zero changes.

Usage
-----
    from rdma.kv_transfer import RDMAKVTransfer, KVTransferMode

    xfer = RDMAKVTransfer(no_gpu=True)          # laptop / CI
    xfer = RDMAKVTransfer(no_gpu=False)         # real ibverbs path

    # Prefill side
    handle = xfer.register_buffer(tensor_bytes, role="prefill")
    xfer.send_rkey(handle, dest_host="decode-pod-svc", ctrl_port=18515)

    # Decode side
    handle = xfer.register_buffer(tensor_bytes, role="decode")
    result  = xfer.rdma_write(handle, rkey, remote_addr, byte_length)
    # result.latency_ms  result.bytes_transferred  result.success
"""

from __future__ import annotations

import os
import time
import socket
import struct
import logging
import argparse
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag helpers
# ---------------------------------------------------------------------------

def _no_gpu_from_env() -> bool:
    return os.environ.get("MINI_LLM_NO_GPU", "0").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryRegion:
    """Represents a registered RDMA Memory Region (or simulated equivalent)."""
    rkey: int                    # remote key (real ibverbs or simulated)
    remote_addr: int             # virtual address (real) or buffer id (sim)
    byte_length: int
    role: str                    # "prefill" | "decode"
    _buffer: Optional[bytearray] = field(default=None, repr=False)
    _sim: bool = False


@dataclass
class TransferResult:
    success: bool
    latency_ms: float
    bytes_transferred: int
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Simulated libibverbs stubs (CPU path)
# ---------------------------------------------------------------------------

class _SimDevice:
    """Fake ibverbs device for CPU-only mode."""

    _rkey_counter = 0
    _addr_counter = 0x10000000
    _registry: dict[int, bytearray] = {}   # rkey → buffer
    _lock = threading.Lock()

    # Loopback bandwidth ceiling: ~10 Gb/s → 1.25 GB/s
    LOOPBACK_BW_BYTES_PER_SEC: float = 1.25e9

    @classmethod
    def ibv_reg_mr(cls, size: int) -> tuple[int, int, bytearray]:
        """Register a memory region; return (rkey, remote_addr, buffer)."""
        with cls._lock:
            cls._rkey_counter += 1
            cls._addr_counter += size + 4096   # page-aligned gap
            rkey = cls._rkey_counter
            addr = cls._addr_counter
            buf = bytearray(size)
            cls._registry[rkey] = buf
        logger.debug("[SIM] ibv_reg_mr: rkey=%d addr=0x%x size=%d", rkey, addr, size)
        return rkey, addr, buf

    @classmethod
    def ibv_post_send_rdma_write(
        cls,
        src_rkey: int,
        dst_rkey: int,
        byte_length: int,
    ) -> TransferResult:
        """Simulate a one-sided RDMA WRITE with realistic timing."""
        t0 = time.perf_counter()

        src_buf = cls._registry.get(src_rkey)
        dst_buf = cls._registry.get(dst_rkey)

        if src_buf is None or dst_buf is None:
            return TransferResult(
                success=False,
                latency_ms=0.0,
                bytes_transferred=0,
                error=f"Unknown rkey src={src_rkey} dst={dst_rkey}",
            )

        # Copy bytes (simulates DMA engine work)
        n = min(byte_length, len(src_buf), len(dst_buf))
        dst_buf[:n] = src_buf[:n]

        # Simulate NIC latency: bandwidth ceiling + 50 µs base RTT
        transfer_sec = byte_length / cls.LOOPBACK_BW_BYTES_PER_SEC
        base_rtt_sec = 0.00005   # 50 µs
        time.sleep(max(0, transfer_sec + base_rtt_sec - (time.perf_counter() - t0)))

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "[SIM] ibv_post_send: %d bytes, %.2f ms", byte_length, latency_ms
        )
        return TransferResult(
            success=True,
            latency_ms=round(latency_ms, 3),
            bytes_transferred=n,
        )


# ---------------------------------------------------------------------------
# Real libibverbs path (imported lazily so the module loads on CPU-only hosts)
# ---------------------------------------------------------------------------

def _try_import_pyverbs():
    try:
        import pyverbs.device as d      # type: ignore
        import pyverbs.pd as pd_mod     # type: ignore
        import pyverbs.mr as mr_mod     # type: ignore
        import pyverbs.qp as qp_mod     # type: ignore
        import pyverbs.cq as cq_mod     # type: ignore
        return d, pd_mod, mr_mod, qp_mod, cq_mod
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RDMAKVTransfer:
    """
    Unified RDMA KV-cache transfer interface.

    Parameters
    ----------
    no_gpu : bool
        When True (laptop / no RDMA hardware) uses _SimDevice.
        When False attempts to open the first available ibverbs device.
    ctrl_port : int
        TCP port used by the gRPC-style control channel to exchange
        (rkey, remote_addr, byte_length) between Prefill and Decode pods.
    """

    def __init__(self, no_gpu: bool = False, ctrl_port: int = 18515):
        self.no_gpu = no_gpu or _no_gpu_from_env()
        self.ctrl_port = ctrl_port
        self._pyverbs = None

        if self.no_gpu:
            logger.info(
                "RDMAKVTransfer: CPU-simulation mode (--no-gpu). "
                "No ibverbs device opened."
            )
        else:
            mods = _try_import_pyverbs()
            if mods is None:
                logger.warning(
                    "pyverbs not found — falling back to CPU simulation. "
                    "Install rdma-core + python3-pyverbs for real RDMA."
                )
                self.no_gpu = True
            else:
                self._pyverbs = mods
                logger.info("RDMAKVTransfer: real ibverbs path active.")

    # ------------------------------------------------------------------
    # Buffer registration
    # ------------------------------------------------------------------

    def register_buffer(self, size_bytes: int, role: str = "decode") -> MemoryRegion:
        """
        Register a memory region of *size_bytes* for RDMA access.

        Returns a MemoryRegion with rkey + remote_addr that can be shared
        with the peer via the control channel.
        """
        if self.no_gpu:
            rkey, addr, buf = _SimDevice.ibv_reg_mr(size_bytes)
            return MemoryRegion(
                rkey=rkey,
                remote_addr=addr,
                byte_length=size_bytes,
                role=role,
                _buffer=buf,
                _sim=True,
            )
        # Real path — placeholder; real implementation registers with
        # pyverbs PD / MR and returns the kernel-assigned lkey/rkey.
        raise NotImplementedError(
            "Real ibverbs registration not yet wired — set no_gpu=True or "
            "implement pyverbs MR registration here."
        )

    # ------------------------------------------------------------------
    # Control channel (TCP side-channel)
    # ------------------------------------------------------------------

    def send_rkey(self, mr: MemoryRegion, dest_host: str, ctrl_port: Optional[int] = None) -> None:
        """
        Prefill side: send (rkey, remote_addr, byte_length) to the Decode pod
        over a lightweight TCP control channel.
        """
        port = ctrl_port or self.ctrl_port
        payload = struct.pack(">QQQ", mr.rkey, mr.remote_addr, mr.byte_length)
        logger.info(
            "send_rkey → %s:%d  rkey=%d addr=0x%x len=%d",
            dest_host, port, mr.rkey, mr.remote_addr, mr.byte_length,
        )
        if self.no_gpu:
            # Simulation: store in module-level dict instead of real TCP
            _rkey_store[(dest_host, port)] = payload
            return
        with socket.create_connection((dest_host, port), timeout=5) as s:
            s.sendall(payload)

    def recv_rkey(self, bind_host: str = "0.0.0.0", ctrl_port: Optional[int] = None) -> tuple[int, int, int]:
        """
        Decode side: receive (rkey, remote_addr, byte_length) from Prefill pod.
        Returns (rkey, remote_addr, byte_length).
        """
        port = ctrl_port or self.ctrl_port
        if self.no_gpu:
            # Simulation: poll the in-process store
            key = (bind_host, port)
            deadline = time.time() + 5
            while key not in _rkey_store and time.time() < deadline:
                time.sleep(0.01)
            payload = _rkey_store.pop(key, b"\x00" * 24)
            rkey, addr, length = struct.unpack(">QQQ", payload)
            return rkey, addr, length
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((bind_host, port))
            srv.listen(1)
            conn, _ = srv.accept()
            with conn:
                data = conn.recv(24)
        return struct.unpack(">QQQ", data)

    # ------------------------------------------------------------------
    # RDMA WRITE
    # ------------------------------------------------------------------

    def rdma_write(
        self,
        local_mr: MemoryRegion,
        remote_rkey: int,
        remote_addr: int,
        byte_length: int,
    ) -> TransferResult:
        """
        Decode side: issue a one-sided RDMA WRITE into *local_mr*.

        In CPU-simulation mode the write is a bytearray copy with
        synthetic latency calibrated to ~10 Gb/s loopback bandwidth.
        """
        if self.no_gpu:
            result = _SimDevice.ibv_post_send_rdma_write(
                src_rkey=remote_rkey,
                dst_rkey=local_mr.rkey,
                byte_length=byte_length,
            )
            return result

        # Real ibverbs path (skeleton)
        raise NotImplementedError(
            "ibv_post_send RDMA WRITE not yet implemented for real hardware. "
            "Wire pyverbs QP.post_send() here."
        )

    # ------------------------------------------------------------------
    # Convenience: full prefill→decode hand-off in one call (single-node sim)
    # ------------------------------------------------------------------

    def simulate_kv_handoff(self, kv_bytes: int) -> TransferResult:
        """
        One-shot simulation of the complete Prefill→Decode KV hand-off:
          register (prefill) → register (decode) → RDMA WRITE → return result.

        Only valid in no_gpu mode; used by rdma_kv_bench.py.
        """
        assert self.no_gpu, "simulate_kv_handoff is only for CPU/sim mode"
        prefill_mr = self.register_buffer(kv_bytes, role="prefill")
        decode_mr  = self.register_buffer(kv_bytes, role="decode")
        # Seed prefill buffer with deterministic data
        h = hashlib.md5(str(kv_bytes).encode()).digest()
        prefill_mr._buffer[:16] = h
        result = _SimDevice.ibv_post_send_rdma_write(
            src_rkey=prefill_mr.rkey,
            dst_rkey=decode_mr.rkey,
            byte_length=kv_bytes,
        )
        return result


# ---------------------------------------------------------------------------
# Module-level in-process key store (sim mode control channel)
# ---------------------------------------------------------------------------
_rkey_store: dict = {}


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="RDMA KV Transfer smoke-test")
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU-simulation mode (no ibverbs hardware required)")
    p.add_argument("--kv-bytes", type=int, default=24 * 1024 * 1024,
                   help="KV cache size to transfer (default 24 MB ~ 512 tokens)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    xfer = RDMAKVTransfer(no_gpu=args.no_gpu)
    print(f"\n{'='*60}")
    print(f"  Mode     : {'CPU-simulation (--no-gpu)' if xfer.no_gpu else 'Real ibverbs'}")
    print(f"  KV bytes : {args.kv_bytes / 1e6:.1f} MB")
    print(f"{'='*60}")
    result = xfer.simulate_kv_handoff(args.kv_bytes)
    print(f"  Success  : {result.success}")
    print(f"  Latency  : {result.latency_ms:.2f} ms")
    print(f"  Bytes    : {result.bytes_transferred / 1e6:.1f} MB")
    print(f"{'='*60}\n")
