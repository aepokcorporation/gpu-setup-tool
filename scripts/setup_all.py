#!/usr/bin/env python3
import os
import sys
import json
import argparse
import yaml
from utils import log_info, log_error, load_session, save_session, reset_progress

def run_step(cmd):
    import subprocess
    try:
        log_info(f"Executing step: {cmd}")
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False

def record_progress(step):
    state = {"last_successful_step": step}
    if not os.path.exists("logs"):
        os.makedirs("logs")
    with open("logs/progress.json", "w") as f:
        json.dump(state, f)

def get_last_successful_step():
    if os.path.exists("logs/progress.json"):
        with open("logs/progress.json") as f:
            state = json.load(f)
            return state.get("last_successful_step", 0)
    return 0

def generate_dockerfile():
    dockerfile_content = """
    FROM nvidia/cuda:11.8.0-base-ubuntu20.04
    RUN apt-get update && apt-get install -y python3 python3-pip && \
        pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
    COPY . /workspace
    WORKDIR /workspace
    CMD ["python3", "scripts/validate_gpu.py"]
    """
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    log_info("Dockerfile generated successfully.")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate full GPU setup.")
    parser.add_argument("--no-frameworks", action="store_true", help="Skip framework installation.")
    parser.add_argument("--frameworks", nargs="+", help="Specify frameworks to install.")
    parser.add_argument("--preset", type=str, help="Use a predefined preset from configs/presets.yaml")
    parser.add_argument("--docker", action="store_true", help="Enable Docker container setup.")
    parser.add_argument("--singularity", action="store_true", help="Enable Singularity container setup.")
    args = parser.parse_args()

    if args.docker:
        log_info("Docker setup selected. Generating Dockerfile...")
        generate_dockerfile()
        sys.exit(0)  # Exit after generating Dockerfile for containerized workflows.

    frameworks_override = []
    if args.preset:
        with open("configs/presets.yaml") as f:
            presets = yaml.safe_load(f)
        preset = presets["presets"].get(args.preset)
        if not preset:
            log_error(f"Preset '{args.preset}' not found.")
            sys.exit(1)
        frameworks_override = preset.get("frameworks", [])

    final_frameworks = []
    if frameworks_override:
        final_frameworks.extend(frameworks_override)
    if args.frameworks:
        final_frameworks.extend(args.frameworks)

    fw_cmd = "python3 scripts/install_frameworks.py"
    if args.no_frameworks:
        fw_cmd += " --no-frameworks"
    elif final_frameworks:
        fw_cmd += " --frameworks " + " ".join(final_frameworks)

    steps = [
        ("Detection", "python3 scripts/detection.py"),
        ("Install GPU Drivers", "python3 scripts/install_gpu_drivers.py"),
        ("Install CUDA & Libraries", "python3 scripts/install_cuda.py")
    ]
    if not args.no_frameworks:
        steps.append(("Install Frameworks", fw_cmd))

    steps.append(("Validation", "python3 scripts/validate_gpu.py"))
    steps.append(("Benchmark", "python3 tests/benchmark.py"))

    log_info("Starting full setup process...")
    last_step = get_last_successful_step()

    for i, (step_name, command) in enumerate(steps, start=1):
        if i <= last_step:
            log_info(f"Skipping {step_name}, already completed.")
            continue

        print(f"=== Running: {step_name} ===")
        if run_step(command):
            record_progress(i)
            log_info(f"{step_name} completed successfully.")
        else:
            log_error(f"{step_name} failed.")
            print(f"{step_name} failed. Check logs/error_log.txt for details.")
            sys.exit(1)

    print("All steps completed successfully. Your GPU environment is ready!")
    print("Check logs/validation_log.txt for framework test results and performance metrics.")
    if os.path.exists("logs/validation_log.txt"):
        with open("logs/validation_log.txt") as f:
            lines = f.read().splitlines()
            summary_lines = [l for l in lines if "successful" in l.lower() or "shape" in l.lower()]
            if summary_lines:
                print("Validation Highlights:")
                for line in summary_lines:
                    print(" -", line)

if __name__ == "__main__":
    main()
