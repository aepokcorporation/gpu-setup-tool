import os
import sys
import json
import traceback
import datetime
from subprocess import CalledProcessError, check_call

LOG_DIR = "logs"
ERROR_LOG = os.path.join(LOG_DIR, "error_log.txt")
INSTALL_SESSION_LOG = os.path.join(LOG_DIR, "install_session.json")
PROGRESS_FILE = os.path.join(LOG_DIR, "progress.json")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

# -------------------------------
# Docker/Singularity Helper Functions
# -------------------------------
def generate_dockerfile(cuda_version, frameworks):
    """
    Generate a Dockerfile for GPU containerization.
    """
    ensure_log_dir()
    dockerfile_content = f"""
    FROM nvidia/cuda:{cuda_version}-base-ubuntu20.04

    # Install dependencies
    RUN apt-get update && apt-get install -y python3 python3-pip && \\
        apt-get clean && rm -rf /var/lib/apt/lists/*

    # Install frameworks
    """
    for framework in frameworks:
        dockerfile_content += f"RUN pip3 install {framework}\n"

    dockerfile_content += """
    # Set up environment
    ENV PATH="/usr/local/cuda/bin:$PATH"
    ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

    CMD ["nvidia-smi"]
    """
    
    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    log_info(f"Dockerfile generated at {dockerfile_path}")

def build_docker_image(image_name="gpu-setup-tool"):
    """
    Build a Docker image from the generated Dockerfile.
    """
    try:
        log_info(f"Building Docker image: {image_name}")
        safe_subprocess_call(["docker", "build", "-t", image_name, "."])
    except Exception as e:
        log_error(f"Failed to build Docker image: {e}", sys.exc_info())
        raise RuntimeError(f"Failed to build Docker image: {e}")

def run_docker_container(image_name="gpu-setup-tool", command="nvidia-smi"):
    """
    Run a Docker container from the built image.
    """
    try:
        log_info(f"Running Docker container: {image_name}")
        safe_subprocess_call(["docker", "run", "--gpus", "all", image_name, command])
    except Exception as e:
        log_error(f"Failed to run Docker container: {e}", sys.exc_info())
        raise RuntimeError(f"Failed to run Docker container: {e}")

# -------------------------------
# Existing Core Functions
# -------------------------------

def load_session():
    if os.path.exists(INSTALL_SESSION_LOG):
        with open(INSTALL_SESSION_LOG) as f:
            return json.load(f)
    return {"apt_packages": [], "pip_packages": [], "steps_completed": []}

def save_session(session):
    ensure_log_dir()
    with open(INSTALL_SESSION_LOG, "w") as f:
        json.dump(session, f, indent=2)

def log_info(message):
    ensure_log_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[INFO] {message}")
    with open(os.path.join(LOG_DIR, "install_log.txt"), "a") as f:
        f.write(f"[INFO {timestamp}] {message}\n")

def log_error(message, exc_info=None):
    ensure_log_dir()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ERROR] {message}")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[ERROR {timestamp}] {message}\n")
        if exc_info:
            traceback.print_exception(*exc_info, file=f)

def safe_subprocess_call(cmd, retries=1, show_progress=False, total_steps=1):
    for attempt in range(retries + 1):
        try:
            log_info(f"Running command: {' '.join(cmd)} (Attempt {attempt+1}/{retries+1})")
            if TQDM_AVAILABLE and show_progress:
                with tqdm(total=total_steps, desc="Running Command", unit="step") as pbar:
                    check_call(cmd)
                    pbar.update(total_steps)
            else:
                check_call(cmd)
            return True
        except CalledProcessError as e:
            log_error(f"Command failed on attempt {attempt+1}: {' '.join(cmd)}", sys.exc_info())
            if attempt >= retries:
                log_error(f"Exhausted retries for command: {' '.join(cmd)}")
                return False
        except FileNotFoundError as e:
            log_error(f"Command not found: {' '.join(cmd)}", sys.exc_info())
            return False
    return False

def record_apt_package(pkg):
    session = load_session()
    if pkg not in session["apt_packages"]:
        session["apt_packages"].append(pkg)
    save_session(session)

def record_pip_package(pkg):
    session = load_session()
    if pkg not in session["pip_packages"]:
        session["pip_packages"].append(pkg)
    save_session(session)

def record_step_completion(step_number):
    session = load_session()
    if step_number not in session["steps_completed"]:
        session["steps_completed"].append(step_number)
    save_session(session)

def reset_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

def rollback(level="all"):
    """
    Rollback partial installations.
    `level` can be "all", "apt-only", or "pip-only".
    """
    log_info(f"Initiating rollback: level={level}")
    session = load_session()

    try:
        if level in ["all", "pip-only"]:
            for pip_pkg in reversed(session["pip_packages"]):
                log_info(f"Uninstalling pip package: {pip_pkg}")
                safe_subprocess_call(["pip3", "uninstall", "-y", pip_pkg])

        if level in ["all", "apt-only"]:
            for apt_pkg in reversed(session["apt_packages"]):
                log_info(f"Purging apt package: {apt_pkg}")
                safe_subprocess_call(["sudo", "apt-get", "remove", "--purge", "-y", apt_pkg])
    except Exception as e:
        log_error(f"Rollback failed: {e}", sys.exc_info())
    finally:
        # Reset session and progress regardless of success
        session = {"apt_packages": [], "pip_packages": [], "steps_completed": []}
        save_session(session)
        reset_progress()
        log_info("Rollback complete. Session and progress reset.")
