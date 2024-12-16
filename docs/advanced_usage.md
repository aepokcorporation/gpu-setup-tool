# Advanced Usage

## Custom Frameworks and Steps
- `--frameworks pytorch cirq` installs only those frameworks.
- `--no-frameworks` skips frameworks.

## Handling Unknown GPUs or OSes
If your GPU or OS isn’t recognized, the tool uses the `unknown_gpu` or `fallback` configs. To improve support:

1. Open `configs/compatibility.yaml`.
2. Add a new entry with:
   - `driver_version`, `cuda_version`
   - `libraries` (cuDNN, NCCL)
   - `frameworks` (with versions)
   - Optional `expected_performance` metrics for benchmarking comparison.
3. Re-run `setup_all.py`. The tool will detect your new configuration.

By defining these values, you provide the script a “known path” to set up and validate your specific environment, reducing reliance on fallback defaults.

## CI/CD Integration
Integrate `setup_all.py` into CI workflows to pre-configure GPU builds.

## Extending Benchmarks
You can modify `tests/benchmark.py` to add custom models or datasets for more realistic performance tests.
