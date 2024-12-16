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

def load_session():
    if os.path.exists(INSTALL_SESSION_LOG):
        with open(INSTALL_SESSION_LOG) as f:
            return json.load(f)
    return {"apt_packages": [], "pip_packages": [], "steps_completed": []}

def save_session(session):
    with open(INSTALL_SESSION_LOG, "w") as f:
        json.dump(session, f, indent=2)

def log_info(message):
    ensure_log_dir()
    print(f"[INFO] {message}")
    with open(os.path.join(LOG_DIR, "install_log.txt"), "a") as f:
        f.write(f"[INFO {datetime.datetime.now()}] {message}\n")

def log_error(message, exc_info=None):
    ensure_log_dir()
    print(f"[ERROR] {message}")
    with open(ERROR_LOG, "a") as f:
        f.write(f"[ERROR {datetime.datetime.now()}] {message}\n")
        if exc_info:
            traceback.print_exception(*exc_info, file=f)

def safe_subprocess_call(cmd, retries=1, show_progress=False, total_steps=1):
    # If show_progress is True and TQDM_AVAILABLE, show a progress bar for steps.
    for attempt in range(retries + 1):
        try:
            log_info(f"Running command: {' '.join(cmd)} (Attempt {attempt+1}/{retries+1})")
            if TQDM_AVAILABLE and show_progress:
                # Simulate progress with tqdm if possible
                with tqdm(total=total_steps, desc="Installing", unit="step") as pbar:
                    # Just run the command once here, increment the bar at the end.
                    check_call(cmd)
                    pbar.update(total_steps)
            else:
                check_call(cmd)
            return True
        except CalledProcessError:
            if attempt < retries:
                log_error(f"Command failed, retrying: {' '.join(cmd)}", sys.exc_info())
            else:
                log_error(f"Command failed after {retries+1} attempts: {' '.join(cmd)}", sys.exc_info())
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

def rollback():
    log_info("Attempting rollback of partial installations...")
    session = load_session()

    # Uninstall pip packages
    for pip_pkg in reversed(session["pip_packages"]):
        log_info(f"Uninstalling pip package: {pip_pkg}")
        safe_subprocess_call(["pip3", "uninstall", "-y", pip_pkg])

    # Purge apt packages
    for apt_pkg in reversed(session["apt_packages"]):
        log_info(f"Purging apt package: {apt_pkg}")
        safe_subprocess_call(["sudo", "apt-get", "remove", "--purge", "-y", apt_pkg])

    # Reset session and progress
    session = {"apt_packages": [], "pip_packages": [], "steps_completed": []}
    save_session(session)
    reset_progress()

    log_info("Rollback complete. Session and progress reset.")
