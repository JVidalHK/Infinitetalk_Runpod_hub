"""Records which GPU this worker actually got, and whether SageAttention runs on it.

Two failures made this necessary, and both were invisible without it.

RunPod schedules serverless workers by price tier, not by GPU model. The 48 GB
tier pools L40, L40S, RTX 6000 Ada and a 48 GB MIG slice carved out of an RTX
PRO 6000; the 80 GB tier pools three H100 variants whose memory bandwidth spans
nearly 2x at one price. Every generation benchmark taken here was attributed to
a card nobody had verified, and at least one "L40S" result turned out to be a
MIG partition — half of a different GPU from a different generation.

The base images ship SageAttention compiled for a single architecture. A worker
that lands on a foreign one does not fall back to a slower kernel; it raises on
the first attention call, ComfyUI produces no output node, and the handler
reports only "video not found" with no hint that the hardware was the cause.

Probing once at startup turns both into data. The result lands in
/gpu_info.json: the entrypoint reads it to decide whether ComfyUI may use sage,
and the handler returns it with every job so no measurement is ever again
attributed to a GPU that was never confirmed.

Deliberately its own process, run before ComfyUI. A missing-kernel error
poisons the CUDA context for the rest of the process that hit it, so the probe
has to be able to fail without taking the server down with it.
"""

import json

INFO_PATH = "/gpu_info.json"


def probe():
    info = {"sage_ok": False, "sage_error": None}

    try:
        import torch
    except Exception as e:
        info["error"] = f"torch unavailable: {type(e).__name__}: {e}"
        return info

    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda

    try:
        if not torch.cuda.is_available():
            info["error"] = "cuda unavailable"
            return info
        props = torch.cuda.get_device_properties(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = f"sm_{major}{minor}"
        info["vram_mb"] = round(props.total_memory / (1024 * 1024))
        # A MIG slice reports a fraction of the memory of the card it was carved
        # from, which is the difference between benchmarking a GPU and
        # benchmarking part of one.
        info["is_mig"] = "MIG" in info["device_name"]
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info

    # Importing sageattention proves the package is installed. It does not prove
    # its compiled kernels cover this architecture — only running one does.
    try:
        from sageattention import sageattn

        q, k, v = (
            torch.randn(1, 4, 64, 64, device="cuda", dtype=torch.float16)
            for _ in range(3)
        )
        sageattn(q, k, v)
        torch.cuda.synchronize()
        info["sage_ok"] = True
    except Exception as e:
        info["sage_error"] = f"{type(e).__name__}: {e}"[:300]

    return info


if __name__ == "__main__":
    result = probe()
    with open(INFO_PATH, "w") as f:
        json.dump(result, f)
    print(f"[gpu_probe] {json.dumps(result)}", flush=True)
