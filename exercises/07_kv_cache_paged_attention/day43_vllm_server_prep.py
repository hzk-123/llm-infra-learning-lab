from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def package_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "README.md").exists() and (parent / "scripts").exists():
            return parent
    return current.parents[2]


def main() -> None:
    project_root = find_project_root()
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    served_model_name = os.environ.get("SERVED_MODEL_NAME", "qwen2.5-0.5b-instruct")
    port = os.environ.get("PORT", "8000")
    max_model_len = os.environ.get("MAX_MODEL_LEN", "2048")
    gpu_memory_utilization = os.environ.get("GPU_MEMORY_UTILIZATION", "0.85")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    hf_home = os.environ.get("HF_HOME", "~/llm-sprint/model_cache/huggingface")

    print("System:")
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    print("project_root:", project_root)

    print("\nRequired tools:")
    rows = [
        ["torch", package_installed("torch")],
        ["vllm", package_installed("vllm")],
        ["openai", package_installed("openai")],
        ["curl", shutil.which("curl") is not None],
        ["tmux", shutil.which("tmux") is not None],
        ["nvidia-smi", shutil.which("nvidia-smi") is not None],
    ]
    print_table(["tool", "available"], rows)

    print("\nServer config:")
    config_rows = [
        ["MODEL_ID", model_id],
        ["SERVED_MODEL_NAME", served_model_name],
        ["PORT", port],
        ["MAX_MODEL_LEN", max_model_len],
        ["GPU_MEMORY_UTILIZATION", gpu_memory_utilization],
        ["CUDA_VISIBLE_DEVICES", cuda_visible_devices],
        ["HF_HOME", hf_home],
    ]
    print_table(["name", "value"], config_rows)

    print("\nStart in tmux:")
    print("conda activate llm-sprint")
    print(f"cd {project_root}")
    print("tmux new -s vllm-day43")
    print("bash scripts/start_vllm_server.sh 2>&1 | tee logs/08_serving_vllm/day43_vllm_server.log")

    print("\nHealth check from another SSH session:")
    print("conda activate llm-sprint")
    print(f"cd {project_root}")
    print(f"curl http://127.0.0.1:{port}/v1/models")
    print("nvidia-smi")

    print("\nOverride examples:")
    print("MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct SERVED_MODEL_NAME=qwen2.5-1.5b-instruct bash scripts/start_vllm_server.sh")
    print("CUDA_VISIBLE_DEVICES=1 PORT=8001 bash scripts/start_vllm_server.sh")
    print("MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.90 bash scripts/start_vllm_server.sh")

    print("\nMeaning:")
    print("Day43 prepares a reproducible vLLM server startup command.")
    print("Use a small model first to separate service issues from performance issues.")
    print("The real smoke test is: server starts, GPU memory is used, and /v1/models responds.")


if __name__ == "__main__":
    main()

