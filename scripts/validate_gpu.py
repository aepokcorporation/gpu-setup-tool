#!/usr/bin/env python3
import os
import sys
import subprocess
import json
from utils import log_info, log_error

def run_tensorflow_test():
    # More robust: Load a small model (e.g., ResNet50) and run inference on GPU
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import resnet50
        model = resnet50.ResNet50(weights=None)  # minimal model, random init
        model = model.to_gpu = True
        with tf.device('/GPU:0'):
            dummy_input = tf.random.normal([1,224,224,3])
            pred = model(dummy_input)
        return f"TensorFlow ResNet inference on GPU successful. Output shape: {pred.shape}"
    except Exception as e:
        return f"TensorFlow test failed: {e}"

def run_pytorch_test():
    # Multi-layer MLP forward pass
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
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
        return f"PyTorch MLP forward pass on GPU successful. Output shape: {y.shape}"
    except Exception as e:
        return f"PyTorch test failed: {e}"

def run_qiskit_test():
    try:
        import qiskit
        qc = qiskit.QuantumCircuit(2)
        qc.h(0)
        qc.cx(0,1)
        sim = qiskit.Aer.get_backend('aer_simulator')
        result = qiskit.execute(qc, sim, shots=1024).result()
        counts = result.get_counts()
        return f"Qiskit test successful, counts: {counts}"
    except Exception as e:
        return f"Qiskit test failed: {e}"

def run_cirq_test():
    try:
        import cirq
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1))
        sim = cirq.Simulator()
        result = sim.run(circuit, repetitions=10)
        return f"Cirq test successful, results: {result}"
    except Exception as e:
        return f"Cirq test failed: {e}"

def validate_gpu():
    try:
        output = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        return True, output
    except:
        return False, "nvidia-smi check failed."

def test_cuda():
    try:
        output = subprocess.check_output(["nvcc", "--version"], universal_newlines=True)
        return True, output
    except:
        return False, "CUDA nvcc not found."

def main():
    gpu_ok, gpu_msg = validate_gpu()
    cuda_ok, cuda_msg = test_cuda()

    tf_result = run_tensorflow_test()
    pt_result = run_pytorch_test()
    qk_result = run_qiskit_test()
    cq_result = run_cirq_test()

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

    if gpu_ok and cuda_ok:
        log_info("Validation successful! GPU and CUDA are properly configured.")
        print("Validation successful!")
    else:
        log_error("Validation encountered issues.")
        print("Validation issues encountered. Check logs/validation_log.txt and logs/error_log.txt.")

if __name__ == "__main__":
    main()
