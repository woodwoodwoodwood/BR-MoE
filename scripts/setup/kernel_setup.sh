#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../BR-MoE/kernels"
python setup_brmoe.py install
