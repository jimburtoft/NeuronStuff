#!/usr/bin/env python3
"""Cross-vendor NIXL transfer test (Neuron HBM <-> CUDA HBM) over LIBFABRIC/EFA.

One process per host. The "target" exposes a registered buffer; the "initiator"
READs from it or WRITEs into it. Metadata + descriptors are exchanged over a plain
TCP socket (no etcd).

  target:    python xvendor_xfer.py target    --mem neuron --port 18515
  initiator: python xvendor_xfer.py initiator --mem cuda --peer <target_ip> --port 18515 --op READ

--mem: neuron | cuda | cpu   (cpu => DRAM segment)
"""
import argparse
import os
import pickle
import socket
import struct
import sys
import time

import torch

SIZES = [4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20, 256 << 20]


def send_obj(s, o):
    b = pickle.dumps(o)
    s.sendall(struct.pack("!Q", len(b)) + b)


def recv_obj(s):
    n = struct.unpack("!Q", _recvn(s, 8))[0]
    return pickle.loads(_recvn(s, n))


def _recvn(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            raise ConnectionError("peer closed")
        buf += c
    return buf


def device_for(mem, dev):
    if mem == "neuron":
        import libtorch_neuronx_lite  # noqa: F401  registers privateuseone
        return torch.device(f"privateuseone:{dev}")
    if mem == "cuda":
        return torch.device(f"cuda:{dev}")
    return torch.device("cpu")


def pattern(nbytes, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


def make_agent(name, backends):
    from nixl._api import nixl_agent, nixl_agent_config
    cfg = nixl_agent_config(True, False, 0, backends=backends)
    return nixl_agent(name, cfg)


def alloc_register(agent, mem, dev, nbytes, fill, backends):
    d = device_for(mem, dev)
    t = fill.to(d) if fill is not None else torch.zeros(nbytes, dtype=torch.uint8).to(d)
    seg = "DRAM" if mem == "cpu" else "VRAM"
    devid = 0 if mem == "cpu" else dev
    reg = agent.get_reg_descs([(t.data_ptr(), nbytes, devid, "")], seg)
    agent.register_memory(reg, backends=backends)
    xd = agent.get_xfer_descs([(t.data_ptr(), nbytes, devid)], seg)
    return t, reg, xd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["target", "initiator"])
    ap.add_argument("--mem", required=True, choices=["neuron", "cuda", "cpu"])
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--peer", default=None)
    ap.add_argument("--port", type=int, default=18515)
    ap.add_argument("--op", default="READ", choices=["READ", "WRITE"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--sizes", default=None, help="comma list of bytes")
    a = ap.parse_args()
    backends = ["LIBFABRIC"]
    sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else SIZES
    # Initialize the device runtime BEFORE creating the NIXL agent. If libnrt is not
    # initialized when the LIBFABRIC rails are opened, Neuron VRAM registration silently
    # falls back to FI_HMEM_SYSTEM ("provider does not support FI_HMEM").
    if a.mem != "cpu":
        torch.zeros(1).to(device_for(a.mem, a.dev))
    agent = make_agent(f"{a.role}-{socket.gethostname()}", backends)
    print(f"[{a.role}] agent up mem={a.mem} dev={a.dev} torch={torch.__version__}", flush=True)

    if a.role == "target":
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", a.port))
        srv.listen(1)
        s, addr = srv.accept()
        print(f"[target] peer {addr}", flush=True)
        peer_meta, op, sizes = recv_obj(s)
        peer_name = agent.add_remote_agent(peer_meta)
        for i, n in enumerate(sizes):
            src = pattern(n, 1000 + i)
            t, reg, xd = alloc_register(agent, a.mem, a.dev, n,
                                        src if op == "READ" else None, backends)
            send_obj(s, (agent.get_agent_metadata(), agent.get_serialized_descs(xd)))
            msg = recv_obj(s)  # initiator says done
            ok = None
            if op == "WRITE":
                exp = pattern(n, 2000 + i)
                ok = bool(torch.equal(t.cpu(), exp))
            send_obj(s, ok)
            print(f"[target] size={n} op={op} initiator={msg} target_verify={ok}", flush=True)
            agent.deregister_memory(reg, backends=backends)
            del t
        s.close()
        return

    s = socket.create_connection((a.peer, a.port))
    send_obj(s, (agent.get_agent_metadata(), a.op, sizes))
    print(f"{'size':>10} {'op':>5} {'GB/s':>8} {'us/xfer':>10} verify", flush=True)
    for i, n in enumerate(sizes):
        meta, rdesc_b = recv_obj(s)
        rname = agent.add_remote_agent(meta)
        rdesc = agent.deserialize_descs(rdesc_b)
        fill = pattern(n, 2000 + i) if a.op == "WRITE" else None
        t, reg, ld = alloc_register(agent, a.mem, a.dev, n, fill, backends)
        times, err = [], None
        for it in range(a.iters):
            h = agent.initialize_xfer(a.op, ld, rdesc, rname, b"")
            t0 = time.perf_counter()
            st = agent.transfer(h)
            while True:
                st = agent.check_xfer_state(h)
                if st in ("DONE", "ERR"):
                    break
            dt = time.perf_counter() - t0
            agent.release_xfer_handle(h)
            if st == "ERR":
                err = "ERR"
                break
            times.append(dt)
        ok = None
        if err is None and a.op == "READ":
            ok = bool(torch.equal(t.cpu(), pattern(n, 1000 + i)))
        send_obj(s, err or "DONE")
        tok = recv_obj(s)
        if a.op == "WRITE":
            ok = tok
        if times:
            best = sorted(times)[len(times) // 2]
            print(f"{n:>10} {a.op:>5} {n / best / 1e9:8.2f} {best * 1e6:10.1f} {ok}", flush=True)
        else:
            print(f"{n:>10} {a.op:>5} {'ERR':>8} {'-':>10} {ok}", flush=True)
        agent.deregister_memory(reg, backends=backends)
        agent.remove_remote_agent(rname)
        del t
    s.close()


if __name__ == "__main__":
    sys.exit(main())
