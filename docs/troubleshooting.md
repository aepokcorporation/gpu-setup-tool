## TROUBLESHOOTING

If automatic recovery fails during setup, follow these steps to resolve common issues.

## STEP 1: CHECK LOGS

1. Review `logs/error_log.txt` for details on what went wrong.
2. Check `logs/install_log.txt` for a step-by-step record of what was completed.

## STEP 2: UPDATE YOUR SYSTEM
Sometimes, missing dependencies or outdated packages can cause issues. 

Run the following commands to update your system:
```bash
sudo apt-get update && sudo apt-get upgrade -y
```

Then, re-run the setup script:
```bash
python3 scripts/setup_all.py
```

## STEP 3: CUSTOM GPU OR OS COMPATIBILITY
If your GPU or OS isn’t recognized:

1. Follow the guide in `docs/advanced_usage.md` to extend compatibility by editing `configs/compatibility.yaml`.
2. Add a new entry with the following details:
  - driver_version and cuda_version
  - Libraries such as cuDNN and NCCL
  - Frameworks with specific versions
  - Optional `expected_performance` metrics for benchmarking comparison
3. After defining these values, re-run the setup script to apply the changes.

## STEP 4: DOCKER/SINGULARITY ISSUES
If using Docker or Singularity:

Ensure docker is installed and running:
```bash
docker --version
```

For Singularity, verify it’s installed and working:
```bash
singularity --version
```

Check that your Dockerfile or container recipe was generated correctly:

- For Docker: Ensure Dockerfile exists and matches your configuration.
- For Singularity: Verify the .def file if applicable.

## STEP 5: CLOUD-SPECIFIC ISSUES
For AWS, Azure, or GCP setups:

1. Ensure GPU drivers are installed:
```bash
nvidia-smi
```

2. Verify GPU quotas and instance types:
  - AWS: Use NVIDIA DL AMIs for pre-configured environments.
  - Azure: Use NC-series or ND-series VMs.
  - GCP: Ensure your project has sufficient GPU quotas.

## STEP 6: OPEN AN ISSUE
If none of the above resolves the issue:

Open a GitHub issue in the repository:
https://github.com/aepokcorporation/gpu-setup-tool/issues

Include:
- Your `logs/error_log.txt` file.
- Steps you’ve already tried.
- Details about your GPU, OS, and setup.

### We’ll do our best to assist you!!
