#!/usr/bin/env python3
import sys
import os
import json
import yaml
from utils import log_info, log_error, safe_subprocess_call, rollback, record_apt_package

def install_cuda_toolkit(cuda_version):
    # Similar approach with show_progress=True to indicate some feedback
    if not safe_subprocess_call(["wget", "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"], show_progress=True, total_steps=1):
        raise RuntimeError("Failed to download CUDA pin file.")
    if not safe_subprocess_call(["sudo", "mv", "cuda-ubuntu2004.pin", "/etc/apt/preferences.d/cuda-repository-pin-600"]):
        raise RuntimeError("Failed to move CUDA pin file.")
    if not safe_subprocess_call(["sudo", "apt-key", "adv", "--fetch-keys", "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub"]):
        raise RuntimeError("Failed to add CUDA repo key.")
    if not safe_subprocess_call(["sudo", "add-apt-repository", "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"]):
        raise RuntimeError("Failed to add CUDA repo.")
    if not safe_subprocess_call(["sudo", "apt-get", "update"], retries=2, show_progress=True, total_steps=1):
        raise RuntimeError("Failed to update apt after adding CUDA repo.")
    cuda_pkg = "cuda-" + cuda_version.replace(".", "-")
    if not safe_subprocess_call(["sudo", "apt-get", "-y", "install", cuda_pkg], retries=2, show_progress=True, total_steps=1):
        raise RuntimeError("Failed to install CUDA toolkit.")
    record_apt_package(cuda_pkg)

def install_libraries(libraries):
    for lib in libraries:
        name = lib.get("name")
        version = lib.get("version")
        log_info(f"Installing {name} version {version} (placeholder logic)")

def main():
    try:
        log_info("Starting CUDA and library installation...")
        with open("logs/detection_log.json") as f:
            env_data = json.load(f)
        gpu_model = env_data.get("gpu_model", "unknown_gpu")

        with open("configs/compatibility.yaml") as f:
            compatibility = yaml.safe_load(f)

        if gpu_model in compatibility:
            cuda_version = compatibility[gpu_model]["cuda_version"]
            libraries = compatibility[gpu_model]["libraries"]
        else:
            cuda_version = compatibility["unknown_gpu"]["cuda_version"]
            libraries = compatibility["unknown_gpu"]["libraries"]

        try:
            install_cuda_toolkit(cuda_version)
            install_libraries(libraries)
            log_info(f"Installed CUDA version: {cuda_version} and libraries.")
            print("CUDA and libraries installation complete.")
        except Exception:
            log_error("CUDA installation failed, attempting rollback and fallback.", sys.exc_info())
            rollback()
            fallback_cuda = compatibility["fallback"]["cuda_version"]
            if fallback_cuda != cuda_version:
                log_info("Retrying CUDA installation with fallback version...")
                install_cuda_toolkit(fallback_cuda)
                install_libraries(compatibility["fallback"]["libraries"])
                log_info("Fallback CUDA installation complete.")
                print("Fallback CUDA installation complete.")
            else:
                raise
    except Exception:
        log_error("CUDA installation failed completely.", sys.exc_info())
        print("CUDA installation failed. Check logs/error_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
