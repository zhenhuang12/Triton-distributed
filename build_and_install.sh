#!/bin/bash
export TRITON_BUILD_WITH_CLANG_LLD=TRUE
export TRITON_USE_ASSERT_ENABLED_LLVM=TRUE
export TRITON_BUILD_PROTON=0
export TRITON_DIST_SHMEM_BACKEND=mori_shmem

rm -f /usr/local/bin/cmake
apt-get update -y
apt install -y libopenmpi-dev git cython3 ibverbs-utils openmpi-bin libopenmpi-dev libpci-dev libdw1 locales cmake miopen-hip autoconf libtool flex ninja-build clang lld
python3 -m pip install -i https://test.pypi.org/simple hip-python>=7.1 # (or whatever Rocm version you have)
pip3 install pybind11

pip3 install -e python --verbose --no-build-isolation --use-pep517