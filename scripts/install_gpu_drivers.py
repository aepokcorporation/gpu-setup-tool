#!/usr/bin/env python3
import sys
import os
import json
import yaml
from utils import log_info, log_error, safe_subprocess_call, rollback, record_apt_package

def install_driver(driver_version):
    major_ver = driver_version.split('.')[0]
    pkg = f"nvidia-driver-{major_ver}"
    # Show progress bar for this step as a single-step process:
    if not safe_subprocess_call(["sudo", "apt-get", "update"], retries=2, show_progress=True, total_steps=1):
        raise RuntimeError("Failed to update package list.")
    # Installing the driver with a progress bar:
    if not safe_subprocess_call(["sudo", "apt-get", "-y", "install", pkg], retries=2, show_progress=True, total_steps=1):
        raise RuntimeError("Driver installation failed.")
    record_apt_package(pkg)

def main():
    try:
        log_info("Starting GPU driver installation...")
        with open("logs/detection_log.json") as f:
            env_data = json.load(f)
        gpu_model = env_data.get("gpu_model", "unknown_gpu")

        with open("configs/compatibility.yaml") as f:
            compatibility = yaml.safe_load(f)

        driver_version = compatibility.get(gpu_model, compatibility["unknown_gpu"])["driver_version"]

        try:
            install_driver(driver_version)
            log_info(f"Installed NVIDIA driver version: {driver_version}")
            print("GPU driver installation complete.")
        except Exception:
            log_error("GPU driver installation failed, attempting rollback and fallback.", sys.exc_info())
            rollback()
            fallback_driver = compatibility["fallback"]["driver_version"]
            if fallback_driver != driver_version:
                log_info("Retrying with fallback driver...")
                install_driver(fallback_driver)
                log_info("Fallback driver installed successfully.")
                print("Fallback GPU driver installation complete.")
            else:
                raise
    except Exception:
        log_error("GPU driver installation failed completely.", sys.exc_info())
        print("GPU driver installation failed. Check logs/error_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
