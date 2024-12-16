#!/usr/bin/env python3
import sys
import os
import json
import yaml
import argparse
from utils import log_info, log_error, safe_subprocess_call, rollback, record_pip_package, record_apt_package

def ensure_pip():
    safe_subprocess_call(["sudo", "apt-get", "update"], retries=2)
    if safe_subprocess_call(["sudo", "apt-get", "-y", "install", "python3-pip"], retries=2, show_progress=True, total_steps=1):
        record_apt_package("python3-pip")
    else:
        raise RuntimeError("Failed to ensure pip is installed.")

def pip_install(package):
    if safe_subprocess_call(["pip3", "install", package], retries=2, show_progress=True, total_steps=1):
        record_pip_package(package.split("==")[0].split(">=")[0])
    else:
        raise RuntimeError(f"Failed to install {package} via pip.")

def install_pytorch(version, cuda_version):
    cu_tag = "cu" + cuda_version.replace(".", "")
    pkg = f"torch=={version}"
    extras = ["torchvision", "torchaudio", "--extra-index-url", f"https://download.pytorch.org/whl/{cu_tag}"]
    cmd = ["pip3", "install", pkg] + extras
    if safe_subprocess_call(cmd, retries=2, show_progress=True, total_steps=1):
        record_pip_package("torch")
        record_pip_package("torchvision")
        record_pip_package("torchaudio")
    else:
        raise RuntimeError("PyTorch installation failed.")

def install_tensorflow(version):
    pkg = f"tensorflow=={version}"
    pip_install(pkg)

def install_qiskit(version):
    pkg = "qiskit" if version == "latest" else f"qiskit=={version}"
    pip_install(pkg)

def install_cirq(version):
    pkg = "cirq" if version == "latest" else f"cirq=={version}"
    pip_install(pkg)

def main():
    parser = argparse.ArgumentParser(description="Install ML and quantum frameworks.")
    parser.add_argument("--frameworks", nargs="+", default=None,
                        help="Specify which frameworks to install.")
    parser.add_argument("--no-frameworks", action="store_true", help="Skip framework installation.")
    args = parser.parse_args()

    if args.no_frameworks:
        log_info("Skipping frameworks installation as requested.")
        sys.exit(0)

    log_info("Starting framework installations...")
    ensure_pip()

    with open("logs/detection_log.json") as f:
        env_data = json.load(f)
    gpu_model = env_data.get("gpu_model", "unknown_gpu")

    with open("configs/compatibility.yaml") as f:
        compatibility = yaml.safe_load(f)

    fw_versions = compatibility.get(gpu_model, compatibility["unknown_gpu"])["frameworks"]
    cuda_version = compatibility.get(gpu_model, compatibility["unknown_gpu"])["cuda_version"]

    if args.frameworks:
        frameworks_to_install = [fw.lower() for fw in args.frameworks]
    else:
        frameworks_to_install = list(fw_versions.keys())

    try:
        for fw in frameworks_to_install:
            ver = fw_versions.get(fw, "latest")
            if fw == "pytorch":
                install_pytorch(ver, cuda_version)
            elif fw == "tensorflow":
                install_tensorflow(ver)
            elif fw == "qiskit":
                install_qiskit(ver)
            elif fw == "cirq":
                install_cirq(ver)
            else:
                log_info(f"Unknown framework: {fw}, skipping.")
        print("Framework installations complete.")
    except Exception:
        log_error("Framework installation failed, attempting rollback.", sys.exc_info())
        rollback()
        fallback_fw_versions = compatibility["fallback"]["frameworks"]
        if fallback_fw_versions != fw_versions:
            log_info("Retrying framework installation with fallback versions...")
            for fw in frameworks_to_install:
                ver = fallback_fw_versions.get(fw, "latest")
                if fw == "pytorch":
                    install_pytorch(ver, compatibility["fallback"]["cuda_version"])
                elif fw == "tensorflow":
                    install_tensorflow(ver)
                elif fw == "qiskit":
                    install_qiskit(ver)
                elif fw == "cirq":
                    install_cirq(ver)
            log_info("Fallback framework installation complete.")
            print("Fallback framework installations complete.")
        else:
            print("Framework installation failed even after fallback. Check logs.")
            sys.exit(1)

if __name__ == "__main__":
    main()
