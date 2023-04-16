
#include <iostream>
#include <vector>
#include <random>
#include <Kokkos_Core.hpp>

#include <ArborX.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Random.hpp>

#include <type_traits>

#include "sim_config.hpp"
#include "Element.hpp"
#include "Quaternion.hpp"
#include "smath.hpp"
#include "World.hpp"

using namespace std;


int main(int argc, char *argv[]){
    Kokkos::ScopeGuard guard(argc, argv);
    using ExecutionSpace = Kokkos::Cuda;
    using CudaMemorySpace = ExecutionSpace::memory_space;
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostMemorySpace = HostExecutionSpace::memory_space;

    ExecutionSpace execution_space{};
    HostExecutionSpace host_execution_space{};

    cout<<execution_space.name()<<endl;

    return 0;
}