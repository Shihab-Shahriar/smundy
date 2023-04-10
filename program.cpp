
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

template <>
struct ArborX::AccessTraits<Kokkos::View<ArborX::Box *, Kokkos::CudaSpace> , ArborX::PrimitivesTag>
{
  using memory_space = typename Kokkos::CudaSpace;
  static KOKKOS_FUNCTION int size(Kokkos::View<ArborX::Box *, Kokkos::CudaSpace>  const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION ArborX::Box const &get(Kokkos::View<ArborX::Box *, Kokkos::CudaSpace>  const &boxes, int i)
  {
    return boxes(i);
  }
};


template <>
struct ArborX::AccessTraits<Kokkos::View<ArborX::Box *, Kokkos::CudaSpace> , ArborX::PredicatesTag>
{
  using memory_space = typename Kokkos::CudaSpace;
  static KOKKOS_FUNCTION int size(Kokkos::View<ArborX::Box *, Kokkos::CudaSpace>  const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Kokkos::View<ArborX::Box *, Kokkos::CudaSpace>  const &boxes, int i)
  {
    return attach(intersects(boxes(i)), (int)i);
  }
};


struct ExcludeSelfCollision
{
  template <class Predicate, class OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i,
                                  OutputFunctor const &out) const
  {
    int const j = ArborX::getData(predicate);
    if (i != j)
    {
      out(i);
    }
  }
};


int main(int argc, char *argv[]){
    Kokkos::ScopeGuard guard(argc, argv);
    using ExecutionSpace = Kokkos::Cuda;
    using CudaMemorySpace = ExecutionSpace::memory_space;
    using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
    using HostMemorySpace = HostExecutionSpace::memory_space;
    
    ExecutionSpace execution_space{};
    HostExecutionSpace host_execution_space{};

    const Configuration CONFIG;
    //World world;

    const int N = 1000;

    Kokkos::View<Sphere *, CudaMemorySpace> spheres(
        Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                        "Sphere::aabb"), N);

    Kokkos::Random_XorShift64_Pool<CudaMemorySpace> random_pool(12345);

    std::cout<<"Allocated Views"<<endl;

    Kokkos::RangePolicy<ExecutionSpace> rpolicy(0,N);
    Kokkos::parallel_for("inits", rpolicy, KOKKOS_LAMBDA(int i){
        Kokkos::Random_XorShift64<CudaMemorySpace> gen = random_pool.get_state();
        random_init(spheres(i), gen, CONFIG);
        random_pool.free_state(gen);
    });
    std::cout<<"XXXXXXXXXXXXXX"<<endl;


    Kokkos::View<ArborX::Box *, CudaMemorySpace> boxes("Example::boxes",N);
    
    auto boxes_host = Kokkos::create_mirror(boxes);

    // std::cout<<particle_host.is_hostspace<<endl;
    // std::cout<<particles.is_hostspace<<endl;


    Kokkos::parallel_for("init_parts", rpolicy, KOKKOS_LAMBDA(int i){
        boxes(i) = get_aabb(spheres(i));
    });

    std::cout<<"3XXXXXXXXXXXXXX"<<endl;


    Kokkos::deep_copy(boxes_host, boxes);
    Kokkos::fence();

    auto spheres_host = Kokkos::create_mirror_view(spheres);
    Kokkos::deep_copy(spheres_host, spheres);
    // for(int i=0;i<N;i++){
    //     std::cout<< spheres_host(i).coords[0]<<","<<spheres_host(i).coords[1]<<","<<spheres_host(i).coords[2]<<endl;
    // }

    ArborX::BVH<CudaMemorySpace> index(execution_space, boxes);

    Kokkos::View<int *, CudaMemorySpace> indices("Example::indices", 0);
    Kokkos::View<int *, CudaMemorySpace> offsets("Example::offsets", 0);
    index.query(execution_space, boxes, ExcludeSelfCollision{}, indices, offsets);
    execution_space.fence();


    Kokkos::View<Linker *, CudaMemorySpace> linkers(
    Kokkos::view_alloc(execution_space, Kokkos::WithoutInitializing,
                    "Linker::aabb"), indices.extent(0));  //we're allocating twice the needed memory

    generate_collision_constraints(spheres, linkers, indices, offsets);
    compute_center_of_mass_force_and_torque(spheres, linkers);
    Kokkos::fence();


    std::cout << "Starting checking results." << '\n';
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto indices_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
    auto linkers_host = 
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, linkers);

    Kokkos::fence();
    std::cout << "Copied Views to host" << '\n';

    // for(int i=0;i<N;i++){
    //     for (int j = offsets_host(i); j < offsets_host(i + 1); ++j)
    //     {
    //       printf("%i %i\n", i, indices_host(j));
    //     }
    // }

    for(int i=0;i<10;i++){
        Linker linker = linkers_host(i);
        printf("%f %f %f\n", linker.signed_sep_dist, linker.lagrange_multiplier, linker.constraint_attachment_locs[0]);
    }

    std::cout<<offsets.extent(0)<<endl;
    std::cout<<indices.extent(0)<<endl;

    return 0;
}