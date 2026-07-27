#!/usr/bin/env bash

set -e

# For local builds, using --jobs 4 is recommended to avoid excessive resource use.
# Keep parallelism caller-configurable here so CI and dedicated builders can choose appropriately.

VERSION=
DEBUG=
PROFILE=
JOBS=
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG=1;;
        --profile) PROFILE=1;;
        --limit-parallel) JOBS=2;;
        -j|--jobs|--parallel)
            OPT="$1"
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Error: ${OPT} requires a positive integer" >&2
                exit 1
            fi
            JOBS="$1"
            ;;
        -j*) JOBS="${1#-j}";;
        --jobs=*) JOBS="${1#*=}";;
        --parallel=*) JOBS="${1#*=}";;
        --limit-parallel=*) JOBS="${1#*=}";;
        35) VERSION="35";;
        36) VERSION="36";;
        37) VERSION="37";;
        38) VERSION="38";;
        39) VERSION="39";;
        310) VERSION="310";;
        311) VERSION="311";;
        312) VERSION="312";;
        313) VERSION="313";;
    esac
    shift
done

if [ -n "${JOBS}" ]; then
  if ! [[ "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: parallel jobs must be a positive integer, got '${JOBS}'" >&2
    exit 1
  fi
  export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
fi

[ -z $VERSION ] && echo "Version not specified, building for all versions" || echo "Compile for Python ${VERSION}"
( [ $DEBUG ] && echo "Debug Mode" ) || ( [ $PROFILE ] && echo "Profile Mode" )  || echo "Release Mode"
[ -n "${JOBS}" ] && echo "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"

function build_sapien() {
  echo "Building SAPIEN"
  BIN=/opt/python/cp310-cp310/bin/python
  COMMAND="${BIN} setup.py bdist_wheel"
  [ $PROFILE ] && COMMAND="${BIN} setup.py bdist_wheel --profile"
  [ $DEBUG ] && COMMAND="${BIN} setup.py bdist_wheel --debug"
  eval "${COMMAND} --sapien-only --build-dir=docker_sapien_build"
}

function build_pybind() {
  echo "Building Pybind"

  PY_VERSION=$1
  if [ "$PY_VERSION" -eq 35 ]; then
      PY_DOT=3.5
      EXT="m"
  elif [ "$PY_VERSION" -eq 36 ]; then
      PY_DOT=3.6
      EXT="m"
  elif [ "$PY_VERSION" -eq 37 ]; then
      PY_DOT=3.7
      EXT="m"
  elif [ "$PY_VERSION" -eq 38 ]; then
      PY_DOT=3.8
      EXT=""
  elif [ "$PY_VERSION" -eq 39 ]; then
      PY_DOT=3.9
      EXT=""
  elif [ "$PY_VERSION" -eq 310 ]; then
      PY_DOT=3.10
      EXT=""
  elif [ "$PY_VERSION" -eq 311 ]; then
      PY_DOT=3.11
      EXT=""
  elif [ "$PY_VERSION" -eq 312 ]; then
      PY_DOT=3.12
      EXT=""
  elif [ "$PY_VERSION" -eq 313 ]; then
      PY_DOT=3.13
      EXT=""
  else
    echo "Error, python version not found!"
  fi

  INCLUDE_PATH=/opt/python/cp${PY_VERSION}-cp${PY_VERSION}${EXT}/include/python${PY_DOT}${EXT}
  BIN=/opt/python/cp${PY_VERSION}-cp${PY_VERSION}${EXT}/bin/python
  export CPLUS_INCLUDE_PATH=${INCLUDE_PATH}
  COMMAND="${BIN} setup.py bdist_wheel"
  eval "${COMMAND} --pybind-only --build-dir=docker_sapien_build"

  PACKAGE_VERSION=`${BIN} setup.py --get-version`
  # Wheel metadata normalizes numeric PEP 440 local versions by removing leading zeros.
  # Resolve the same normalized version before locating the wheel for auditwheel.
  PACKAGE_VERSION=`${BIN} -c "from packaging.version import Version; print(Version('${PACKAGE_VERSION}'))"`
  WHEEL_NAME="./dist/sapien-${PACKAGE_VERSION}-cp${PY_VERSION}-cp${PY_VERSION}${EXT}-linux_x86_64.whl"
  if test -f "$WHEEL_NAME"; then
    echo "$WHEEL_NAME exists, begin audit and repair"
  fi
  # libcuda.so.1 is the host NVIDIA driver API. It is required by the OIDN CUDA
  # module but must remain external and be supplied by the NVIDIA driver/runtime.
  auditwheel repair ${WHEEL_NAME} --exclude 'libvulkan*' --exclude 'libOpenImageDenoise*' --exclude 'libcuda.so*' --internal libsapien --internal libsvulkan2

  # auditwheel derives the platform tag from versioned glibc symbols alone, so anything glibc
  # exports unversioned is invisible to it. A PhysX build on glibc 2.38+ dragged in
  # __isoc23_strtoul and friends -- unversioned, introduced in 2.38 -- and the wheel still got
  # tagged manylinux_2_28 while failing to import on everything below 2.38. Every other symbol
  # in those libraries topped out at GLIBC_2.2.5, so the tag was the only thing that lied.
  #
  # Fail the build rather than ship that again. Keep the inferred tag: it is correct once the
  # unversioned blind spot is empty.
  local repaired
  repaired=$(ls -1 wheelhouse/sapien-${PACKAGE_VERSION}-cp${PY_VERSION}-cp${PY_VERSION}${EXT}-manylinux*_x86_64.whl 2>/dev/null | head -1)
  if [ -z "${repaired}" ]; then
    echo "auditwheel produced no manylinux wheel for cp${PY_VERSION}" >&2
    exit 1
  fi
  ${BIN} - "${repaired}" <<'PYCHECK'
import re, subprocess, sys, zipfile, tempfile, pathlib

wheel = pathlib.Path(sys.argv[1])
# Unversioned glibc symbols that pin a floor auditwheel cannot see. __isoc23_* arrived in 2.38.
suspect = re.compile(r"__isoc23_\w+")
found = {}
with tempfile.TemporaryDirectory() as tmp:
    with zipfile.ZipFile(wheel) as zf:
        shared = [n for n in zf.namelist() if ".so" in n]
        zf.extractall(tmp, members=shared)
    for name in shared:
        path = pathlib.Path(tmp) / name
        out = subprocess.run(["nm", "-D", "--undefined-only", str(path)],
                             capture_output=True, text=True).stdout
        hits = sorted(set(suspect.findall(out)))
        if hits:
            found[name] = hits

if found:
    print(f"{wheel.name} carries unversioned glibc symbols the platform tag does not cover:",
          file=sys.stderr)
    for name, hits in found.items():
        print(f"  {name}: {', '.join(hits)}", file=sys.stderr)
    print("Rebuild the dependency on an older glibc; the tag would understate the real floor.",
          file=sys.stderr)
    sys.exit(1)

print(f"{wheel.name}: no unversioned glibc floor beyond the inferred tag")
PYCHECK
}

build_sapien
if [ -z "${VERSION}" ]
then
   build_pybind 39
   build_pybind 310
   build_pybind 311
   build_pybind 312
   build_pybind 313
else
   build_pybind $VERSION
fi
