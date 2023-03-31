export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64:/usr/local/cuda-12.1/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

export KOKKOS_INSTALL_DIR=/home/shihab/src/smundy/build/kokkos