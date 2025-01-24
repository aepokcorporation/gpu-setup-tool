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
