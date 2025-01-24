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

def configure_ldconfig(cuda_path):
    log_info("Configuring ldconfig for CUDA and cuDNN.")
    cuda_lib_path = os.path.join(cuda_path, "lib64")
    with open("/etc/ld.so.conf.d/cuda.conf", "w") as f:
        f.write(cuda_lib_path + "\n")
    os.system("sudo ldconfig")

def configure_env_vars(cuda_path):
    log_info("Configuring environment variables for CUDA and cuDNN.")
    bashrc_path = os.path.expanduser("~/.bashrc")
    with open(bashrc_path, "a") as f:
        f.write(f"\n# CUDA Environment Variables\n")
        f.write(f"export PATH={cuda_path}/bin:$PATH\n")
        f.write(f"export LD_LIBRARY_PATH={cuda_path}/lib64:$LD_LIBRARY_PATH\n")
    os.system("source ~/.bashrc")

def install_cuda_toolkit(cuda_version, cloud_env):
    log_info(f"Installing CUDA Toolkit version {cuda_version} for {cloud_env}.")
    try:
        if cloud_env == "AWS":
            log_info("Skipping CUDA installation on AWS as AMI drivers are pre-installed.")
            return
        elif cloud_env == "Azure" or cloud_env == "GCP":
            log_info("Using cloud-specific repositories for CUDA installation.")
            safe_subprocess_call(["sudo", "apt-get", "update"])
            safe_subprocess_call(["sudo", "apt-get", "-y", "install", f"cuda-{cuda_version}"])
        else:
            # On-Prem installation logic
            if not safe_subprocess_call(["wget", f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-{cuda_version.replace('.', '-')}.deb"]):
                raise RuntimeError("Failed to download CUDA installer.")
            safe_subprocess_call(["sudo", "dpkg", "-i", f"cuda-{cuda_version.replace('.', '-')}.deb"])
            safe_subprocess_call(["sudo", "apt-get", "update"])
            safe_subprocess_call(["sudo", "apt-get", "-y", "install", f"cuda-{cuda_version}"])
        configure_ldconfig(f"/usr/local/cuda-{cuda_version}")
        configure_env_vars(f"/usr/local/cuda-{cuda_version}")
        log_info(f"CUDA Toolkit {cuda_version} installation completed successfully.")
    except Exception as e:
        raise RuntimeError(f"CUDA installation failed: {e}")

def install_cudnn(cuda_version, cudnn_version, cloud_env):
    log_info(f"Installing cuDNN version {cudnn_version} for CUDA {cuda_version} on {cloud_env}.")
    try:
        if cloud_env == "AWS" or cloud_env == "Azure":
            log_info("Skipping cuDNN installation as it is pre-installed on cloud.")
            return
        cudnn_pkg_url = f"https://developer.download.nvidia.com/compute/redist/cudnn/v{cudnn_version}/cudnn-{cuda_version}-linux-x64-v{cudnn_version}.tgz"
        safe_subprocess_call(["wget", cudnn_pkg_url])
        safe_subprocess_call(["tar", "-xvf", f"cudnn-{cuda_version}-linux-x64-v{cudnn_version}.tgz", "-C", "/usr/local"])
        configure_ldconfig(f"/usr/local/cuda-{cuda_version}")
        log_info(f"cuDNN {cudnn_version} installation completed successfully.")
    except Exception as e:
        raise RuntimeError(f"cuDNN installation failed: {e}")

def main():
    try:
        log_info("Starting CUDA and cuDNN installation...")
        with open("logs/detection_log.json") as f:
            env_data = json.load(f)
        gpu_model = env_data.get("gpu_model", "unknown_gpu")

        with open("configs/compatibility.yaml") as f:
            compatibility = yaml.safe_load(f)

        cuda_version = compatibility.get(gpu_model, compatibility["unknown_gpu"])["cuda_version"]
        cudnn_version = compatibility.get(gpu_model, compatibility["unknown_gpu"])["cudnn_version"]

        cloud_env = detect_cloud_environment()

        try:
            install_cuda_toolkit(cuda_version, cloud_env)
            install_cudnn(cuda_version, cudnn_version, cloud_env)
            log_info(f"Installed CUDA version {cuda_version} and cuDNN version {cudnn_version} for {cloud_env}.")
            log_info("CUDA and cuDNN installation completed successfully.")  # New success log entry
            print("CUDA and cuDNN installation complete.")
        except Exception:
            log_error("CUDA installation failed, attempting rollback.", sys.exc_info())
            rollback()
            sys.exit(1)
    except Exception:
        log_error("CUDA installation failed completely.", sys.exc_info())
        print("CUDA installation failed. Check logs/error_log.txt for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
