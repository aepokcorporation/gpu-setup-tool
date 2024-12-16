### Beginner’s Guide  
For a quick setup on a known-good environment:  
```bash  
python3 scripts/setup_all.py --preset tensorflow-ubuntu  
Or just run python3 scripts/setup_all.py and let the tool do the work. It detects your GPU, OS, and tries best-fit configurations, performing rollbacks and retries if needed.

After completion, check logs/validation_log.txt for detailed validation tests (e.g., ResNet inference in TensorFlow).
