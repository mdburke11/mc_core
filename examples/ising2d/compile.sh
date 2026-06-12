#!/bin/bash
set -e

MC_CORE_ROOT=${MC_CORE_ROOT:-$HOME/codes/mc_core/install}

echo "Using MC_CORE_ROOT=$MC_CORE_ROOT"

cmake .. -DMC_CORE_ROOT="$MC_CORE_ROOT" -DCMAKE_BUILD_TYPE=Release
