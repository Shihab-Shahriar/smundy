rm CMakeCache.txt

cmake ../ \
    -DCMAKE_CXX_COMPILER="/home/shihab/src/smundy/dep/Kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_FLAGS="-pg" \
    -DCMAKE_PREFIX_PATH="$KOKKOS_INSTALL_DIR" \
    -DKokkos_ROOT="/home/shihab/src/smundy/dep/Kokkos" \
    -DCMAKE_BUILD_TYPE:STRING=Debug \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ARCH_TURING75=ON 