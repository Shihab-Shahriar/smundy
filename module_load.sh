module purge
module load GCC/10.2.0 CUDA/11.2.1 OpenMPI/4.0.5 netCDF/4.7.4 HDF5/1.10.7 CMake/3.23.1 git OpenBLAS/0.3.13 ParMETIS/4.0.3


export ROOT_DIR=/mnt/gs21/scratch/khanmd/smundy
export Trilinos_Install_DIR=${ROOT_DIR}/install/trilinos
export Trilinos_Source_DIR=${ROOT_DIR}/packages/Trilinos