export TRITON_BUILD_WITH_CLANG_LLD=TRUE
export TRITON_USE_ASSERT_ENABLED_LLVM=TRUE
export TRITON_BUILD_PROTON=0
apt-get update -y
apt install -y libopenmpi-dev git cython3 ibverbs-utils openmpi-bin libopenmpi-dev libpci-dev libdw1 locales cmake miopen-hip autoconf libtool flex ninja-build clang lld
python3 -m pip install -i https://test.pypi.org/simple hip-python>=7.1 # (or whatever Rocm version you have)
pip3 install pybind11
ROCM_ARCH=gfx950 bash ./shmem/rocshmem_bind/build.sh

OMPI_DIR=/opt/ompi \
  ROCM_ARCH=gfx950 \
  TRITON_DIST_SHMEM_BACKEND=rocshmem \
      pip3 install -e python --no-build-isolation --use-pep517