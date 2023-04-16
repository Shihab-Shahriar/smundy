rm CMakeCache.txt

cmake ${ROOT_DIR} \
    -DCMAKE_CXX_COMPILER="${Trilinos_Source_DIR}/packages/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_FLAGS="-pg" \
    -DTrilinos_DIR:PATH=${Trilinos_Install_DIR} \
    -DKokkos_ROOT:PATH="${Trilinos_Install_DIR}" \
    -DCMAKE_BUILD_TYPE:STRING=Debug \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_LAMBDA=ON 