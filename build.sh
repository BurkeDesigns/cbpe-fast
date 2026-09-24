#!/bin/bash
# Development build without maturin: compile the extension and drop it into python/cbpe_fast/,
# so `PYTHONPATH=python python -c "import cbpe_fast"` works. `pip install .` is the packaged route.
set -e
cd "$(dirname "$0")"
PYO3_PYTHON="${PYO3_PYTHON:-$(command -v python3)}" cargo build --release
cp target/release/libcbpe_fast.so python/cbpe_fast/_native.so
echo "built $(pwd)/python/cbpe_fast/_native.so"
