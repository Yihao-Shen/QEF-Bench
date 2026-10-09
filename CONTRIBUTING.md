# Contributing

Open an issue with the affected platform, dependency versions, workload configuration
and a minimal reproduction. Do not attach credentials or private machine details.
For changes, submit a pull request explaining the measurement boundary and validation.
Run `python -B -m unittest discover -s tools -p 'test_*.py'` before submitting.
Hardware-dependent results should state the exact device, clock policy, affinity,
energy boundary, warm-up and sample count. Software-only checks are welcome;
label them separately from measurements. Preserve third-party notices.
