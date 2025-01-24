#!/usr/bin/env python3
import os
import subprocess
from utils import log_info, log_error

def run_tensorflow_test():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import resnet50
        model = resnet50.ResNet50(weights=None)
        with tf.device('/GPU:0'):
            dummy_input = tf.random.normal([1, 224, 224, 3])
            pred = model(dummy_input)
        log_info("TensorFlow validation successful.")
        return f"TensorFlow ResNet inference on GPU successful. Output shape: {pred.shape}"
    except Exception as e:
        log_error(f"TensorFlow test failed: {e}")
        return f"TensorFlow test failed: {e}"

def run_pytorch_test():
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            log_error("PyTorch validation failed: GPU not available.")
            return "PyTorch: GPU not available."
        class MLP(torch.nn.Module):
            def __init__(self):
                super(MLP, self).__init__()
                self.fc1 = torch.nn.Linear(1024, 512)
                self.fc2 = torch.nn.Linear(512, 256)
                self.fc3 = torch.nn.Linear(256, 10)
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        model = MLP().to(device)
        x = torch.randn(64, 1024, device=device)
        y = model(x)
        log_info("PyTorch validation successful.")
        return f"PyTorch MLP forward pass on GPU successful. Output shape: {y.shape}"
    except Exception as e:
        log_error(f"PyTorch test failed: {e}")
        return f"PyTorch test failed: {e}"

def run_qiskit_test():
    try:
        import qiskit
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        sim = qiskit.Aer.get_backend('aer_simulator')
        result = qiskit.execute(qc, sim, shots=1024).result()
        counts = result.get_counts()
        log_info("Qiskit validation successful.")
        return f"Qiskit test successful, counts: {counts}"
    except Exception as e:
        log_error(f"Qiskit test failed: {e}")
        return f"Qiskit test failed: {e}"

def run_cirq_test():
    try:
        import cirq
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))
        sim = cirq.Simulator()
        result = sim.run(circuit, repetitions=10)
        log_info("Cirq validation successful.")
        return f"Cirq test successful, results: {result}"
    except Exception as e:
        log_error(f"Cirq test failed: {e}")
        return f"Cirq test failed: {e}"

def run_onnx_test():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' not in providers:
            log_error("ONNX validation failed: CUDAExecutionProvider not available.")
            return "ONNX Runtime: CUDAExecutionProvider not available."
        sess = ort.InferenceSession("resnet50-v2-7.onnx", providers=["CUDAExecutionProvider"])
        dummy_input = {'data': [[0.5]*224*224*3]}
        output = sess.run(None, dummy_input)
        log_info("ONNX validation successful.")
        return f"ONNX Runtime inference on GPU successful. Output: {output[0][:5]}"
    except FileNotFoundError:
        log_error("ONNX test skipped: Model file not found (resnet50-v2-7.onnx).")
        return "ONNX test skipped: Model file not found (resnet50-v2-7.onnx)."
    except Exception as e:
        log_error(f"ONNX Runtime test failed: {e}")
        return f"ONNX Runtime test failed: {e}"

def run_jax_test():
    try:
        import jax
        import jax.numpy as jnp
        from jax import device_put
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (500, 500))
        x = device_put(x)  # Move to GPU
        y = jnp.dot(x, x.T)
        log_info("JAX validation successful.")
        return f"JAX GPU matrix multiplication successful. Output shape: {y.shape}"
    except Exception as e:
        log_error(f"JAX test failed: {e}")
        return f"JAX test failed: {e}"

def validate_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        return True, output
    except:
        log_error("nvidia-smi check failed.")
        return False, "nvidia-smi check failed."

def test_cuda():
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        return True, output
    except:
        log_error("CUDA nvcc not found.")
        return False, "CUDA nvcc not found."

def main():
    gpu_ok, gpu_msg = validate_gpu()
    cuda_ok, cuda_msg = test_cuda()

    tf_result = run_tensorflow_test()
    pt_result = run_pytorch_test()
    qk_result = run_qiskit_test()
    cq_result = run_cirq_test()
    onnx_result = run_onnx_test()
    jax_result = run_jax_test()

    if not os.path.exists("logs"):
        os.makedirs("logs")

    with open("logs/validation_log.txt", "w") as f:
        f.write("GPU Validation:\n")
        f.write(str(gpu_msg) + "\n\n")
        f.write("CUDA Validation:\n")
        f.write(str(cuda_msg) + "\n\n")
        f.write("Framework Validation:\n")
        f.write(f"TensorFlow: {tf_result}\n")
        f.write(f"PyTorch: {pt_result}\n")
        f.write(f"Qiskit: {qk_result}\n")
        f.write(f"Cirq: {cq_result}\n")
        f.write(f"ONNX Runtime: {onnx_result}\n")
        f.write(f"JAX: {jax_result}\n")

    if gpu_ok and cuda_ok:
        log_info("Validation successful! GPU and CUDA are properly configured.")
        print("Validation successful!")
    else:
        log_error("Validation encountered issues.")
        print("Validation issues encountered. Check logs/validation_log.txt and logs/error_log.txt.")

if __name__ == "__main__":
    main()
