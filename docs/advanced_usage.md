# Advanced Usage

## CUSTOM FRAMEWORKS AND STEPS
Use: 
- `--frameworks` to specify frameworks you want installed. 
For example:
- `--frameworks` pytorch cirq installs only those frameworks.

Use:
 - `--no-frameworks` to skip framework installation entirely.

- These are optional arguments users can pass to the script.
- They provide control over specific parts of the tool's behavior.

## HANDLING UNKNOWN GPUS OR OSES
If your GPU or OS isn’t recognized, the tool defaults to the unknown_gpu or fallback configurations. To improve support for your environment:

1. Open the file configs/compatibility.yaml.
2. Add a new entry with the following details:
- driver_version and cuda_version
- libraries such as cuDNN and NCCL
- frameworks with specific versions
3. Optional expected_performance metrics for benchmarking comparison

After defining these values, re-run setup_all.py. The tool will use your custom configuration to set up and validate your environment, reducing reliance on fallback defaults.

## CI/CD INTEGRATION
You can integrate setup_all.py into CI/CD workflows to pre-configure GPU builds, ensuring consistency across development environments.

## EXTENDING BENCHMARKS
To add custom models or datasets for more realistic performance tests, modify tests/benchmark.py as needed. This allows you to fine-tune benchmarks for your specific workflows.
