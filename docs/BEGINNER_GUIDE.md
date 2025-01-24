### Beginner’s Guide  

For a quick setup on a known-good environment:

```bash
python3 scripts/setup_all.py --preset tensorflow-ubuntu
```

Or just run:
```bash
python3 scripts/setup_all.py
```
The tool detects your GPU, OS, and applies the best-fit configurations. It performs rollbacks and retries if issues occur during the setup.

After completion, check logs/validation_log.txt for detailed validation tests (e.g., ResNet inference in TensorFlow).

### CLONE THE REPOSITORY

To get started, clone the repository to your local machine:
```bash
git clone https://github.com/aepokcorporation/gpu-setup-tool.git
cd gpu-setup-tool
```

### INSTALL PREREQUISITES

Make sure your environment has Python 3.x installed:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip
```

### New Features for Beginners
Docker Setup:
Run the tool in a containerized environment for reproducibility. 

Use:
```bash
python3 scripts/setup_all.py --docker
```
This generates a Dockerfile tailored to your environment.

Cloud Detection:
The tool detects cloud environments (AWS, Azure, GCP) and adjusts installations accordingly.

Expanded Framework Support:
Supports JAX, ONNX Runtime, TensorFlow, PyTorch, and more.
