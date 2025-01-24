# GPU Setup Tool

**Simplify GPU environment configuration, validation, and benchmarking.**  
This tool automates the setup of GPU drivers, CUDA toolkits, libraries (cuDNN, NCCL), and popular frameworks like TensorFlow, PyTorch, JAX, and ONNX Runtime. It detects your system’s GPU and OS, installs everything you need, and validates that your setup works—no hassle required.

---

## Features
- **Automatic Detection:**
  - Identifies your GPU model, OS version, and environment (cloud or local).
- **Seamless Installation:**
  - Installs GPU drivers, CUDA, and libraries with fallback logic for common issues.
- **Framework Support:**
  - Easily set up TensorFlow, PyTorch, JAX, ONNX Runtime, Qiskit, Cirq, and more.
- **Cloud Optimizations:**
  - Tailored configurations for AWS, Azure, and GCP environments.
- **Docker/Singularity Support:**
  - Run containerized workflows for reproducibility and easier debugging.
- **Validation & Benchmarking:**
  - Ensures everything is working with functional tests and performance metrics.

---

## Getting Started

### Requirements
- **Hardware:** NVIDIA GPU (support for AMD coming soon).  
- **OS:** Linux (Ubuntu 20.04+, Debian, CentOS supported).  
- **Optional:** Docker or Singularity for containerized setups.

### Quick Start
1. Clone this repository:
    ```bash
    git clone https://github.com/aepokcorporation/gpu-setup-tool.git
    cd gpu-setup-tool
    ```
2. Run the setup script:
    ```bash
    python3 scripts/setup_all.py
    ```
3. Follow the on-screen instructions. Logs will be saved in the `logs/` directory.

---

## Validation and Benchmarking

Once the setup is complete, the tool runs validation tests to confirm:

1. **Drivers and CUDA Installation:**
   - Checks GPU visibility with `nvidia-smi`.
   - Confirms CUDA compatibility with `nvcc --version`.
2. **Framework Tests:**
   - **TensorFlow:** Runs a ResNet inference test.
   - **PyTorch:** Performs a multi-layer MLP forward pass.
   - **JAX:** Validates matrix multiplication on GPU.
   - **ONNX Runtime:** Tests model inference using a lightweight ONNX model.
3. **Benchmarks:**
   - Measures GPU performance and compares it to expected metrics for known GPUs.

**Logs:** All validation results are saved in `logs/validation_log.txt`.

---

## Contributing

Contributions are welcome! You can help by:
- Reporting unsupported environments or bugs.
- Adding support for new GPUs, frameworks, or operating systems.
- Improving validation or benchmarking scripts.

### How to Contribute
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with your changes.

---

## Roadmap

**Planned Features:**
- Further expand framework support (e.g., TensorRT, PaddlePaddle).
- Enhanced cloud-specific setups for multi-GPU training.
- GUI-based tool for easier user interaction and validation.
- Pre-built Docker images for faster container-based setups.
- AMD GPU support for ROCm environments.

---

## Feedback and Contact

If you encounter issues or have suggestions, feel free to open an issue. Contributions are highly encouraged!

---

## License

This project is licensed under the MIT License.

---

### A Note on Testing

This project is in its early stages and has not been tested across all possible environments.  
If you encounter an issue, please submit it via GitHub issues or contribute directly by forking the repository and submitting a pull request.  

Your feedback and contributions will help make this tool better for everyone!
