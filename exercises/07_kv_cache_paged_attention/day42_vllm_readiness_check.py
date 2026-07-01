from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def package_status(import_name: str, dist_name: str | None = None) -> tuple[str, str]:
    if importlib.util.find_spec(import_name) is None:
        return "missing", "-"

    package_name = dist_name or import_name
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "installed, version unknown"
    return "installed", version


def run_command(args: list[str], timeout: int = 5) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return False, str(exc)

    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def check_torch() -> dict[str, object]:
    info: dict[str, object] = {
        "installed": False,
        "version": "-",
        "cuda_version": "-",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }

    if importlib.util.find_spec("torch") is None:
        return info

    import torch

    info["installed"] = True
    info["version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda or "-"
    info["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        info["device_count"] = count
        devices: list[dict[str, object]] = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gib": props.total_memory / 1024**3,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        info["devices"] = devices

    return info


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "README.md").exists() and (parent / "exercises").exists():
            return parent
    return current.parents[2]


def main() -> None:
    project_root = find_project_root()

    print("System:")
    print("python:", sys.version.replace("\n", " "))
    print("python_executable:", sys.executable)
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("project_root:", project_root)

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("vllm", "vllm"),
        ("openai", "openai"),
        ("requests", "requests"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]

    print("\nPython package check:")
    package_rows = []
    package_map: dict[str, str] = {}
    for import_name, dist_name in packages:
        status, version = package_status(import_name, dist_name)
        package_map[import_name] = status
        package_rows.append([import_name, status, version])
    print_table(["package", "status", "version"], package_rows)

    torch_info = check_torch()
    print("\nTorch / CUDA:")
    torch_rows = [
        ["torch installed", torch_info["installed"]],
        ["torch version", torch_info["version"]],
        ["torch CUDA version", torch_info["cuda_version"]],
        ["CUDA available", torch_info["cuda_available"]],
        ["CUDA device count", torch_info["device_count"]],
    ]
    print_table(["item", "value"], torch_rows)

    devices = torch_info["devices"]
    if devices:
        print("\nGPU devices from torch:")
        device_rows = [
            [
                device["index"],
                device["name"],
                f"{device['total_memory_gib']:.2f} GiB",
                f"{device['major']}.{device['minor']}",
            ]
            for device in devices
        ]
        print_table(["index", "name", "total_memory", "compute_capability"], device_rows)

    print("\nnvidia-smi:")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("nvidia-smi: missing")
    else:
        print("nvidia-smi path:", nvidia_smi)
        ok, output = run_command(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        if ok:
            rows = []
            for line in output.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 4:
                    rows.append([parts[0], parts[1], parts[2] + " MiB", parts[3] + " MiB"])
            print_table(["index", "name", "memory.total", "memory.used"], rows)
        else:
            print("nvidia-smi query failed:", output)

    print("\nModel cache environment:")
    cache_env_names = ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE"]
    cache_rows = [[name, os.environ.get(name, "not set")] for name in cache_env_names]
    print_table(["env", "value"], cache_rows)

    print("\nProject output directories:")
    required_dirs = [
        project_root / "benchmarks",
        project_root / "results",
        project_root / "docs" / "performance_reports",
        project_root / "logs" / "08_serving_vllm",
    ]
    dir_rows = [[str(path.relative_to(project_root)), path.exists()] for path in required_dirs]
    print_table(["path", "exists"], dir_rows)

    cuda_available = bool(torch_info["cuda_available"])
    gpu_count = int(torch_info["device_count"])
    vllm_installed = package_map.get("vllm") == "installed"
    transformers_installed = package_map.get("transformers") == "installed"
    openai_installed = package_map.get("openai") == "installed"

    print("\nReadiness summary:")
    summary_rows = [
        ["local code/dev readiness", package_map.get("torch") == "installed" and transformers_installed],
        ["server CUDA readiness", cuda_available and gpu_count >= 1],
        ["server vLLM readiness", cuda_available and gpu_count >= 1 and vllm_installed],
        ["benchmark client readiness", openai_installed or package_map.get("requests") == "installed"],
    ]
    print_table(["check", "ready"], summary_rows)

    print("\nNext action:")
    if cuda_available and gpu_count >= 1 and vllm_installed:
        print("This machine is ready for Day43 vLLM server command preparation.")
    elif cuda_available and gpu_count >= 1 and not vllm_installed:
        print("CUDA is available, but vLLM is missing. Install vLLM on the server before real serving.")
    elif platform.system().lower() == "windows":
        print("This looks like a Windows dev machine. Use it for code/docs, then run this check on the 4090 server.")
    else:
        print("CUDA/GPU is not ready on this machine. Fix the server environment before starting vLLM.")

    print("\nMeaning:")
    print("Day42 separates environment readiness from model-serving bugs.")
    print("Real benchmark reports should record Python, torch, CUDA, vLLM, GPU, and cache settings.")
    print("Your laptop can be a dev machine; the 3x4090 server should be the real vLLM benchmark machine.")


if __name__ == "__main__":
    main()

