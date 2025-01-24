#!/usr/bin/env python3
import sys
import os
import json
import yaml
import requests
from utils import log_info, log_error, safe_subprocess_call, rollback, record_apt_package

def detect_cloud_environment():
    try:
        # AWS Check
        aws_url = "http://169.254.169.254/latest/meta-data/"
        if requests.get(aws_url, timeout=2).status_code == 200:
            log_info("Detected AWS environment.")
            return "AWS"

        # Azure Check
        azure_url = "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
        headers = {"Metadata": "true"}
        if requests.get(azure_url, headers=headers, timeout=2).status_code == 200:
            log_info("Detected Azure environment.")
            return "Azure"

        # GCP Check
        gcp_url = "http://metadata.google.internal/computeMetadata/v1/"
        headers = {"Metadata-Flavor": "Google"}
        if requests.get(gcp_url, headers=headers, timeout=2).status_code == 200:
            log_info("Detected GCP environment.")
            return "GCP"
    except requests.exceptions.RequestException:
        log_info("No cloud environment detected.")
    return "On-Prem"

def configure_ldconfig(driver_path):
    log_info("Configuring ldconfig for NVIDIA drivers.")
    try:
        with open("/etc/ld.so.conf.d/nvidia.conf", "w") as f:
            f.write(driver_path + "\n")
        os.system("sudo ldconfig")
    except Exception as e:
        raise RuntimeError(f"Failed to configure ldconfig for drivers: {e}")

def configure_env_vars(driver_path):
    log_info("Configuring environment variables for NVIDIA drivers.")
    bashrc_path = os.path.expanduser("~/.bashrc")
    try:
        with open(bashrc_path, "a") as f:
            f.write(f"\n# NVIDIA Driver Environment Variables\n")
            f.write(f"export PATH={driver_path}/bin:$PATH\n")
            f.write(f"export LD_LIBRARY_PATH={driver_path}/lib64:$LD_LIBRARY_PATH\n")
        os.system("source ~/.bashrc")
    except Exception as e:
        raise RuntimeError(f"Failed to configure driver environment variables: {e}")

def install_driver(driver_version, cloud_env):
    log_info(f"Installing NVIDIA driver version {driver_version} for {cloud_env}.")
    try:
        if cloud_env == "AWS":
            log_info("Skipping driver installation on AWS as NVIDIA drivers are pre-installed.")
            return
        elif cloud_env == "Azure" or cloud_env == "GCP":
            log_info("Using cloud-specific repositories for NVIDIA driver installation.")
            safe_subprocess_call(["sudo", "apt-get", "update"])
            safe_subprocess_call(["sudo", "apt-get", "-y", "install", f"nvidia-driver-{driver_version}"])
        else:
            # On-Prem installation logic
            major_ver = driver_version.split(".")[0]
            pkg = f"nvidia-driver-{major_ver}"
            safe_subprocess_call(["sudo", "apt-get", "update"], retries=2)
            safe_subprocess_call(["sudo", "apt-get", "-y", "install", pkg], retries=2)
        configure_ldconfig(f"/usr/lib/nvidia-{driver_version}")
        configure_env_vars(f"/usr/lib/nvidia-{driver_version}")
        record_apt_package(f"nvidia-driver-{driver_version}")
    except Exception as e:
        raise RuntimeError(f"Driver installation failed: {e}")

def main():
    try:
        log_info("Starting GPU driver installation...")
        with open("logs/detection_log.json") as f:
            env_data = json.load(f)
        gpu_model = env_data.get("gpu_model", "unknown_gpu")

        with open("configs/compatibility.yaml") as f:
            compatibility = yaml.safe_load(f)

        driver_version = compatibility.get(gpu_model, compatibility["unknown_gpu"])["driver_version"]
        cloud_env = detect_cloud_environment()

        try:
            install_driver(driver_version, cloud_env)
            log_info(f"Installed NVIDIA driver version: {driver_version} for {cloud_env}.")
            print("GPU driver installation complete.")
        except Exception:
            log_error("Driver installation failed, attempting rollback and fallback.", sys.exc_info())
            rollback()
            fallback_driver = compatibility["fallback"]["driver_version"]
            if fallback_driver != driver_version:
                log_info("Retrying with fallback driver...")
                install_driver(fallback_driver, cloud_env)
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
