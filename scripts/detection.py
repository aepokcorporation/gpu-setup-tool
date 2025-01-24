#!/usr/bin/env python3
import os
import json
import sys
import platform
import subprocess
from utils import log_info, log_error

def detect_gpu():
    try:
        # Use nvidia-smi as the primary method for GPU detection
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], universal_newlines=True)
        gpus = output.strip().splitlines()
        gpu_list = []
        for gpu in gpus:
            gpu_name = gpu.lower().replace(" ", "_")
            if "a100" in gpu_name:
                gpu_list.append("nvidia_a100")
            elif "h100" in gpu_name:
                gpu_list.append("nvidia_h100")
            elif "4090" in gpu_name:
                gpu_list.append("nvidia_rtx_4090")
            elif "3090" in gpu_name:
                gpu_list.append("nvidia_rtx_3090")
            else:
                gpu_list.append("nvidia_generic")
        return gpu_list if gpu_list else ["unknown_gpu"]
    except Exception as e:
        log_error(f"Failed to detect GPU using nvidia-smi: {e}")
        return ["unknown_gpu"]

def detect_os():
    try:
        if os.path.isfile("/etc/os-release"):
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=")[1].strip('"')
        return platform.system()
    except Exception as e:
        log_error(f"Failed to detect OS: {e}")
        return "Unknown OS"

def detect_cloud_provider():
    try:
        # AWS
        aws_url = "http://169.254.169.254/latest/meta-data/instance-id"
        if subprocess.call(["curl", "-s", aws_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return "AWS"
        
        # Azure
        azure_url = "http://169.254.169.254/metadata/instance"
        headers = {"Metadata": "true"}
        if subprocess.call(["curl", "-s", "-H", "Metadata:true", azure_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return "Azure"

        # GCP
        gcp_url = "http://metadata.google.internal/computeMetadata/v1/"
        headers = {"Metadata-Flavor": "Google"}
        if subprocess.call(["curl", "-s", "-H", "Metadata-Flavor:Google", gcp_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return "GCP"

    except Exception as e:
        log_error(f"Failed to detect cloud provider: {e}")
    return "Unknown"

def detect_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        for line in output.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(" ")[0]
                return version
    except Exception as e:
        log_error(f"Failed to detect CUDA version: {e}")
    return "Unknown"

def main():
    try:
        log_info("Starting environment detection...")
        gpu_models = detect_gpu()
        os_name = detect_os()
        cloud_provider = detect_cloud_provider()
        cuda_version = detect_cuda_version()

        if "unknown_gpu" in gpu_models:
            log_info("No supported NVIDIA GPU detected. Using fallback settings.")
        if cloud_provider == "Unknown":
            log_info("No cloud provider detected. Assuming on-prem setup.")

        data = {
            "gpu_models": gpu_models,
            "os": os_name,
            "cloud_provider": cloud_provider,
            "cuda_version": cuda_version
        }

        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open("logs/detection_log.json", "w") as f:
            json.dump(data, f, indent=2)

        log_info(f"Detection complete: {data}")

        # Provide environment-specific suggestions
        if cloud_provider == "Azure":
            log_info("Azure detected. Consider NC-series VMs for optimal GPU performance.")
        elif cloud_provider == "AWS":
            log_info("AWS detected. Use NVIDIA Deep Learning AMIs for pre-configured setups.")
        elif cloud_provider == "GCP":
            log_info("GCP detected. Ensure correct GPU quota and drivers.")

        print("Detection successful. Check logs/detection_log.json for details.")
    except Exception as e:
        log_error("Error during detection", exc_info=sys.exc_info())
        print("Detection failed. Check logs/error_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
