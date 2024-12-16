# GPU Setup Tool

**Simplify GPU environment configuration, validation, and benchmarking.**  
This tool automates the setup of GPU drivers, CUDA toolkits, libraries (cuDNN, NCCL), and popular frameworks like TensorFlow and PyTorch. It detects your system’s GPU and OS, installs everything you need, and validates that your setup works—no hassle required.

---

## Features
- **Automatic Detection:**
  - Identifies your GPU model, OS version, and environment (cloud or local).
- **Seamless Installation:**
  - Installs GPU drivers, CUDA, and libraries with fallback logic for common issues.
- **Framework Support:**
  - Easily set up TensorFlow, PyTorch, Qiskit, Cirq, and more.
- **Validation & Benchmarking:**
  - Ensures everything is working with functional tests and performance metrics.

---

## Getting Started

### Requirements
- **Hardware:** NVIDIA GPU (support for AMD coming soon).  
- **OS:** Linux (Ubuntu 20.04+, Debian, CentOS supported).

### Quick Start
1. Clone this repository:
    ```bash
    git clone https://github.com/exoriantech/gpu-setup-tool.git
    cd gpu-setup-tool
    ```
2. Run the setup script:
    ```bash
    python3 scripts/setup_all.py
    ```
3. Follow the on-screen instructions. Logs will be saved in the `logs/` directory.

### Known Limitations
- Current focus is on NVIDIA GPUs. AMD support is planned for future releases.
- Compatibility may vary across VMs and environments—feedback is welcome to improve the tool.

## Validation and Benchmarking

Once the setup is complete, the tool runs validation tests to confirm:

1. **Drivers and CUDA Installation:**
   - Checks GPU visibility with `nvidia-smi`.
   - Confirms CUDA compatibility with `nvcc --version`.
2. **Framework Tests:**
   - **TensorFlow:** Runs a ResNet inference test.
   - **PyTorch:** Performs a multi-layer MLP forward pass.
3. **Benchmarks:**
   - Measures GPU performance and compares it to expected metrics for known GPUs.

**Logs:** All validation results are saved in `logs/validation_log.txt`.

## Contributing

Contributions are welcome! You can help by:
- Reporting unsupported environments or bugs.
- Adding support for new GPUs, frameworks, or operating systems.
- Improving validation or benchmarking scripts.

### How to Contribute
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with your changes.

## Roadmap

**Planned Features:**
- Docker/Singularity container support for easier reproducibility.
- Expanded framework integration (e.g., JAX, ONNX).
- Optimized configurations for cloud environments (AWS, Azure, GCP).

## Feedback and Contact

If you encounter issues or have suggestions, feel free to open an issue. Contributions are highly encouraged!

## License

This project is licensed under the MIT License.
