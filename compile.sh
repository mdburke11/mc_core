#!/bin/bash
set -e

INSTALL_PREFIX=${INSTALL_PREFIX:-$HOME/codes/mc_core/install}

echo "Installing mc_core to: $INSTALL_PREFIX"

cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" -DCMAKE_BUILD_TYPE=Release
