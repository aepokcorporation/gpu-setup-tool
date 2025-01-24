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
    log_info("Installing PyTorch framework...")
    cu_tag = "cu" + cuda_version.replace(".", "")
    pkg = f"torch=={version}"
    extras = ["torchvision", "torchaudio", "--extra-index-url", f"https://download.pytorch.org/whl/{cu_tag}"]
    cmd = ["pip3", "install", pkg] + extras
    if safe_subprocess_call(cmd, retries=2, show_progress=True, total_steps=1):
        record_pip_package("torch")
        record_pip_package("torchvision")
        record_pip_package("torchaudio")
        log_info("PyTorch installation completed successfully.")
    else:
        raise RuntimeError("PyTorch installation failed.")

def install_tensorflow(version):
    log_info("Installing TensorFlow framework...")
    pkg = f"tensorflow=={version}"
    pip_install(pkg)
    log_info("TensorFlow installation completed successfully.")

def install_jax(cuda_version):
    log_info("Installing JAX framework...")
    if cuda_version == "cpu":
        pip_install("jax[cpu]")
    else:
        pkg = f"https://storage.googleapis.com/jax-releases/jaxlib-0.4.13-{cuda_version}-linux_x86_64.whl"
        pip_install("jax")
        pip_install(pkg)
    log_info("JAX installation completed successfully.")

def install_onnx():
    log_info("Installing ONNX Runtime framework...")
    pip_install("onnxruntime-gpu")
    log_info("ONNX Runtime installation completed successfully.")

def install_qiskit(version):
    log_info("Installing Qiskit framework...")
    pkg = "qiskit" if version == "latest" else f"qiskit=={version}"
    pip_install(pkg)
    log_info("Qiskit installation completed successfully.")

def install_cirq(version):
    log_info("Installing Cirq framework...")
    pkg = "cirq" if version == "latest" else f"cirq=={version}"
    pip_install(pkg)
    log_info("Cirq installation completed successfully.")

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
            elif fw == "jax":
                install_jax(cuda_version)
            elif fw == "onnx":
                install_onnx()
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
                elif fw == "jax":
                    install_jax(compatibility["fallback"]["cuda_version"])
                elif fw == "onnx":
                    install_onnx()
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

