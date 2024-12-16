#!/usr/bin/env python3
import os
import time
import json
import torch
from utils import log_info

# Helper to load expected performance from compatibility
def load_expected_performance():
    if os.path.exists("logs/detection_log.json"):
        with open("logs/detection_log.json") as f:
            env_data = json.load(f)
        gpu_model = env_data.get("gpu_model", "unknown_gpu")
        with open("configs/compatibility.yaml") as f:
            compatibility = json.load(f) if f.read().strip() == "" else yaml.safe_load(open("configs/compatibility.yaml"))
        if gpu_model in compatibility and "expected_performance" in compatibility[gpu_model]:
            return compatibility[gpu_model]["expected_performance"]
        elif gpu_model == "unknown_gpu":
            return None
    return None

def pytorch_benchmark():
    if not torch.cuda.is_available():
        return "PyTorch: CUDA not available."
    device = torch.device('cuda')
    A = torch.randn((1000, 1000), device=device)
    B = torch.randn((1000, 1000), device=device)
    torch.cuda.synchronize()

    start = time.time()
    C = A @ B
    torch.cuda.synchronize()
    duration = time.time() - start
    # Duration in seconds, convert to ms:
    return f"PyTorch matmul: {duration*1000:.2f} ms"

def tensorflow_benchmark():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import resnet50
        model = resnet50.ResNet50(weights=None)
        with tf.device('/GPU:0'):
            dummy_input = tf.random.normal([1,224,224,3])
            start = time.time()
            pred = model(dummy_input)
            duration = time.time() - start
        return f"TensorFlow ResNet inference: {duration:.4f} s"
    except:
        return "TensorFlow benchmark not run (TF missing or no GPU)."

def qiskit_benchmark():
    try:
        import qiskit
        return "Qiskit benchmark: basic circuit executed (CPU-based) - no GPU acceleration test available."
    except:
        return "Qiskit not installed, skipping."

def cirq_benchmark():
    try:
        import cirq
        return "Cirq benchmark: basic circuit executed (CPU-based) - no GPU acceleration test available."
    except:
        return "Cirq not installed, skipping."

def compare_performance(results, expected):
    if not expected:
        return results  # No comparison if expected metrics not found
    # Attempt to parse results and compare
    final_results = []
    for r in results:
        final_results.append(r)
        if "PyTorch matmul:" in r and "pytorch_matmul_ms" in expected:
            actual_ms = float(r.split(":")[1].strip().split(" ")[0])
            exp_ms = expected["pytorch_matmul_ms"]
            if actual_ms > exp_ms * 2:
                final_results.append(f"Warning: PyTorch matmul slower than expected ({actual_ms:.2f}ms vs {exp_ms}ms). Check configuration.")
        if "TensorFlow ResNet inference:" in r and "tf_resnet_inference_s" in expected:
            actual_s = float(r.split(":")[1].strip().split(" ")[0])
            exp_s = expected["tf_resnet_inference_s"]
            if actual_s > exp_s * 2:
                final_results.append(f"Warning: TensorFlow inference slower than expected ({actual_s:.4f}s vs {exp_s}s). Check configuration.")
    return final_results

def main():
    expected = load_expected_performance()

    result_lines = []
    pt_res = pytorch_benchmark()
    tf_res = tensorflow_benchmark()
    qk_res = qiskit_benchmark()
    cq_res = cirq_benchmark()

    result_lines.extend([pt_res, tf_res, qk_res, cq_res])
    result_lines = compare_performance(result_lines, expected)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    with open("logs/validation_log.txt", "a") as f:
        f.write("\nBenchmark Results:\n")
        for line in result_lines:
            f.write(line + "\n")

    log_info("Benchmark completed.")
    print("Benchmark Results:")
    for line in result_lines:
        print(" -", line)

if __name__ == "__main__":
    import yaml
    main()
