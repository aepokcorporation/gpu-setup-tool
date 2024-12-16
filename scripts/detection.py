#!/usr/bin/env python3
import os
import json
import sys
import platform
import subprocess
from utils import log_info, log_error

def detect_gpu():
    # Enhanced detection for unknown GPUs
    try:
        output = subprocess.check_output(["lspci"], universal_newlines=True)
        for line in output.split("\n"):
            if "NVIDIA" in line:
                if "A100" in line.upper():
                    return "nvidia_a100"
                if "RTX 3090" in line.upper():
                    return "nvidia_rtx_3090"
                return "nvidia_generic"
    except:
        pass

    # Try nvidia-smi
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], universal_newlines=True)
        name = output.strip().lower().replace(" ", "_")
        if "a100" in name:
            return "nvidia_a100"
        elif "3090" in name:
            return "nvidia_rtx_3090"
        return "nvidia_generic"
    except:
        pass

    # Check for AMD/Intel as future note
    try:
        output = subprocess.check_output(["lshw", "-C", "display"], universal_newlines=True)
        if "AMD" in output.upper():
            return "amd_unsupported"
        elif "INTEL" in output.upper():
            return "intel_unsupported"
    except:
        pass

    return "unknown_gpu"

def detect_os():
    if os.path.isfile("/etc/os-release"):
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.strip().split("=")[1].strip('"')
    return platform.system()

def detect_cloud_provider():
    # AWS
    try:
        aws_meta = subprocess.check_output(["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-id"], universal_newlines=True)
        if aws_meta:
            return "AWS"
    except:
        pass
    # Azure
    try:
        azure_meta = subprocess.check_output(["curl", "-s", "-H", "Metadata:true", "http://169.254.169.254/metadata/instance?api-version=2021-02-01"], universal_newlines=True)
        if azure_meta:
            return "Azure"
    except:
        pass
    # GCP
    try:
        gcp_meta = subprocess.check_output(["curl", "-s", "http://metadata.google.internal/computeMetadata/v1/instance/id", "-H", "Metadata-Flavor: Google"], universal_newlines=True)
        if gcp_meta:
            return "GCP"
    except:
        pass
    return "Unknown"

def main():
    try:
        log_info("Starting environment detection...")
        gpu_model = detect_gpu()
        os_name = detect_os()
        cloud_provider = detect_cloud_provider()

        if "unsupported" in gpu_model:
            log_info(f"Detected {gpu_model.split('_')[0].upper()} GPU which may not be supported. Will fallback to unknown_gpu settings.")

        if gpu_model not in ["nvidia_a100", "nvidia_rtx_3090", "nvidia_generic"]:
            gpu_model = "unknown_gpu"

        data = {
            "gpu_model": gpu_model,
            "os": os_name,
            "cloud_provider": cloud_provider
        }

        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open("logs/detection_log.json", "w") as f:
            json.dump(data, f, indent=2)

        log_info(f"Detection complete: {data}")

        # Provide suggestions
        if gpu_model == "unknown_gpu":
            log_info("No known NVIDIA GPU detected. Using fallback settings. Check docs/troubleshooting.md if issues arise.")

        if cloud_provider == "Azure":
            log_info("Azure detected. If using NVads VMs (vGPU), performance may be limited. Consider NC-series for full GPU support.")
        elif cloud_provider == "AWS":
            log_info("AWS detected. Consider using NVIDIA DL AMIs for pre-configured environments.")
        elif cloud_provider == "GCP":
            log_info("GCP detected. Ensure you have the right GPU quota and use compatible drivers from GCP repos if needed.")

        print("Detection successful. Check logs/detection_log.json for details.")
    except Exception:
        log_error("Error during detection", exc_info=sys.exc_info())
        print("Detection failed. Check logs/error_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
