rm CMakeCache.txt
rm -rf CMakeFiles

#     -DEigen3_DIR="${ROOT_DIR}/packages/eigen-3.4.0/cmake" \
#     -DCMAKE_PREFIX_PATH="${ROOT_DIR}/packages/eigen-3.4.0" \

cmake ${ROOT_DIR} \
    -DCMAKE_CXX_COMPILER="${Trilinos_Source_DIR}/packages/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_FLAGS="-O3 -lmpi" \
    -DTrilinos_DIR:PATH=${Trilinos_Install_DIR} \
    -DKokkos_ROOT:PATH="${Trilinos_Install_DIR}" \
    -DCMAKE_BUILD_TYPE:STRING=Release \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=OFF \
    -DKokkos_ENABLE_CUDA_LAMBDA=OFF 