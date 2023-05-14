#!/bin/bash
trilinos_src_dir=/mnt/gs21/scratch/khanmd/smundy/packages/Trilinos
build_dir=${trilinos_src_dir}/build/
build_type=${CMAKE_BUILD_TYPE:-Debug}
trilinos_install_dir=${Trilinos_Install_DIR}
fortran_macro=${FORTRAN_MACRO:-FORTRAN_ONE_UNDERSCORE}
#cmake_cxx_flags="-O3 -march=native -lmpi"
cmake_cxx_flags="-lmpi"
cmake_exe_linker_flags=${CMAKE_EXE_LINKER_FLAGS}
cuda_on_or_off=${CUDA:-OFF}
CUDA=$cuda_on_or_off
clear_cache=${CLEAR_CACHE:-ON}

printf "\nTRILINOS_DIR=${trilinos_src_dir}\n";
printf "BUILD_DIR=${build_dir}\n";
printf "CMAKE_BUILD_TYPE=${build_type}\n";
printf "CMAKE_EXE_LINKER_FLAGS=${cmake_exe_linker_flags}\n";
printf "CMAKE_CXX_FLAGS=${cmake_cxx_flags}\n";
printf "CUDA=${cuda_on_or_off}\n";
printf "TRILINOS_INSTALL_DIR=${trilinos_install_dir}\n";
printf "FORTRAN_MACRO=${fortran_macro}\n";
printf "\nTo change these vars, set as env vars or pass to this script like 'VAR=value run_cmake_stk'\n\n";

if [ "${CUDA}" != "OFF" ] && [ "${CUDA}" != "ON" ]; then
  echo "CUDA must be set to 'ON' or 'OFF'"
  exit 1;
fi
not_cuda=ON
if [ "${CUDA}" == "ON" ] ; then
  not_cuda=OFF
fi
printf "not_cuda: ${not_cuda}\n";
if [ ! -d ${trilinos_src_dir}/packages/seacas ] && [ ! -L ${trilinos_src_dir}/packages/seacas ] ; then
  echo "Trilinos dir (${trilinos_src_dir}) doesn't have packages/seacas directory. If using a Sierra project, make a soft-link to Sierra's seacas directory.";
  exit 1;
fi
if [ ! -d ${trilinos_src_dir}/packages/stk ] && [ ! -L ${trilinos_src_dir}/packages/stk ]; then
  echo "Trilinos dir (${trilinos_src_dir}) doesn't have packages/stk directory. If using a Sierra project, make a soft-link to Sierra's stk directory.";
  exit 1;
fi

read

mkdir -p $trilinos_install_dir
mkdir -p $build_dir
cd ${build_dir}
if [ "${clear_cache}" == "ON" ] ; then
# Cleanup old cache before we configure
  rm -rf CMakeFiles CMakeCache.txt
fi


cmake \
-DCMAKE_CXX_COMPILER:FILEPATH="${trilinos_src_dir}/packages/kokkos/bin/nvcc_wrapper" \
-DCMAKE_INSTALL_PREFIX=$trilinos_install_dir \
-DCMAKE_BUILD_TYPE=${build_type^^} \
-DTrilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
-DTrilinos_ENABLE_TESTS:BOOL=ON \
-DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF \
-DTrilinos_ALLOW_NO_PACKAGES:BOOL=OFF \
-DTrilinos_ASSERT_MISSING_PACKAGES=OFF \
-DTPL_ENABLE_MPI=ON \
-DTPL_ENABLE_HDF5=ON \
-DTrilinos_ENABLE_Tpetra:BOOL=ON \
-DTpetraCore_ENABLE_TESTS:BOOL=OFF \
-DTpetra_ENABLE_DEPRECATED_CODE:BOOL=ON \
-DTrilinos_ENABLE_Zoltan2:BOOL=ON \
-DZoltan2_ENABLE_ParMETIS:BOOL=ON \
-DTrilinos_ENABLE_Pamgen:BOOL=ON \
-DTrilinos_ENABLE_Percept:BOOL=ON \
-DTrilinos_ENABLE_Panzer:BOOL=${not_cuda} \
-DTrilinos_ENABLE_PanzerAdaptersSTK:BOOL=${not_cuda} \
-DPanzer_ENABLE_TESTS:BOOL=${not_cuda} \
-DPanzer_STK_ENABLE_TESTS:BOOL=${not_cuda} \
-DTrilinos_ENABLE_TrilinosCouplings:BOOL=${not_cuda} \
-DTPL_ENABLE_CUDA:BOOL=${cuda_on_or_off} \
-DKokkos_ENABLE_CUDA:BOOL=${cuda_on_or_off} \
-DKokkos_ENABLE_CUDA_UVM:BOOL=${cuda_on_or_off} \
-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE:BOOL=OFF \
-DKokkos_ARCH_VOLTA70=${cuda_on_or_off} \
-DTpetra_ENABLE_CUDA:BOOL=${cuda_on_or_off} \
-DTrilinos_ENABLE_KokkosKernels:BOOL=ON \
-DTrilinos_ENABLE_Zoltan:BOOL=ON \
-DTrilinos_ENABLE_Fortran:BOOL=ON \
-DCMAKE_CXX_STANDARD:STRING=17 \
-DCMAKE_CXX_FLAGS:STRING="-DNOT_HAVE_STK_SEACASAPREPRO_LIB -D${fortran_macro} ${cmake_cxx_flags} -Werror=dangling-else" \
-DSTK_ENABLE_TESTS:BOOL=ON \
-DTrilinos_ENABLE_STK:BOOL=ON \
-DTrilinos_ENABLE_Gtest:BOOL=ON \
-DTrilinos_ENABLE_SEACASExodus:BOOL=ON \
-DTrilinos_ENABLE_SEACASIoss:BOOL=ON \
-DTPL_ENABLE_Zlib:BOOL=ON \
-DBLAS_LIBRARY_DIRS:FILEPATH="$EBROOTOPENBLAS/lib" \
-DBLAS_LIBRARY_NAMES:STRING="libopenblas.so" \
-DLAPACK_LIBRARY_DIRS:FILEPATH="$EBROOTOPENBLAS/lib" \
-DLAPACK_LIBRARY_NAMES="libopenblas.so" \
-DTPL_ENABLE_ParMETIS:BOOL=OFF \
-DZoltan2_ENABLE_ParMETIS:BOOL=OFF \
-DHDF5_BASE_DIR:FILEPATH="$HDF5_DIR" \
-DMPI_BASE_DIR:FILEPATH="$EBROOTOPENMPI" \
-DCMAKE_EXE_LINKER_FLAGS="${cmake_exe_linker_flags}" \
${trilinos_src_dir}/