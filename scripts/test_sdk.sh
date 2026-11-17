#!/usr/bin/env bash
# Checks a packed C++ SDK the way a program outside this repository uses it:
#
#   scripts/test_sdk.sh SDK_ARCHIVE [--gpu]
#
# Unpacks SDK_ARCHIVE (dist/sapien-sdk-*.tar.gz from scripts/build.sh) into a temporary directory
# and builds test/sdk twice: through CMake's find_package(sapien), and with the compiler given
# only the SDK's include and library directories. Both run without LD_LIBRARY_PATH on CPU PhysX.
# --gpu also runs them on GPU PhysX (needs an NVIDIA GPU) with an empty home directory, so
# PhysxDefault::EnableGPU() can only find the PhysX GPU runtime the SDK ships.
set -euo pipefail

if [ $# -lt 1 ] || { [ $# -eq 2 ] && [ "$2" != --gpu ]; } || [ $# -gt 2 ]; then
  echo "usage: $0 SDK_ARCHIVE [--gpu]" >&2
  exit 1
fi
ARCHIVE=$(realpath "$1")
SOURCE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../test/sdk" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

tar -C "${WORK}" -xzf "${ARCHIVE}"
SDK=$(ls -d "${WORK}"/sapien-sdk-*)

cmake -S "${SOURCE}" -B "${WORK}/cmake" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${SDK}"
cmake --build "${WORK}/cmake"
# RUNPATH, not RPATH: the program's search path must not cover for the SDK libraries' own.
"${CXX:-g++}" -std=c++20 -O1 -I"${SDK}/include" -I"${SDK}/include/physx/include" \
  "${SOURCE}/main.cpp" -o "${WORK}/plain" \
  -L"${SDK}/lib" -lsapien -Wl,-rpath,"${SDK}/lib" -Wl,--enable-new-dtags

mkdir "${WORK}/home"
for program in "${WORK}/cmake/sdk_test" "${WORK}/plain"; do
  env -u LD_LIBRARY_PATH "${program}" cpu
  if [ $# -eq 2 ]; then
    env -u LD_LIBRARY_PATH HOME="${WORK}/home" "${program}" gpu
  fi
done
echo "SDK test passed: ${ARCHIVE}"
