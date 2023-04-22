#include <vector>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_DynamicView.hpp>

#include <ArborX.hpp>
#include <ArborX_Version.hpp>

#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpAtomics.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/GetNgpField.hpp>

#include "sim_config.hpp"
#include "Element.hpp"
#include "Quaternion.hpp"
#include "smath.hpp"
#include "World.hpp"


struct AABB{
    // ArborX has the constructors declared with 'constexpr', why was that necessary?
    KOKKOS_DEFAULTED_FUNCTION  AABB() = default;

    KOKKOS_INLINE_FUNCTION 
     AABB(ArborX::Box b_, stk::mesh::Entity const &entity_)
        : b(b_), entity(entity_)
    {}
    ArborX::Box b;
    stk::mesh::Entity entity;
};

template <>
struct ArborX::AccessTraits<Kokkos::View<AABB *, Kokkos::CudaSpace> , ArborX::PrimitivesTag>
{
  using memory_space = typename Kokkos::CudaSpace;
  static KOKKOS_FUNCTION int size(Kokkos::View<AABB *, Kokkos::CudaSpace>  const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION ArborX::Box const &get(Kokkos::View<AABB *, Kokkos::CudaSpace>  const &boxes, int i)
  {
    return boxes(i).b;
  }
};


template <>
struct ArborX::AccessTraits<Kokkos::View<AABB *, Kokkos::CudaSpace> , ArborX::PredicatesTag>
{
  using memory_space = typename Kokkos::CudaSpace;
  static KOKKOS_FUNCTION int size(Kokkos::View<AABB *, Kokkos::CudaSpace>  const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Kokkos::View<AABB *, Kokkos::CudaSpace>  const &boxes, int i)
  {
    return attach(intersects(boxes(i).b), (int)i);
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


void generate_neighbor_pairs(const stk::mesh::NgpMesh & ngpMesh,
                              stk::mesh::NgpField<double>& elemAabbField_device, 
                              Kokkos::Experimental::DynamicView<EntityPair*, stk::ngp::MemSpace>& neighbors,
                              int N){

    typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
    const auto& teamPolicy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngpMesh.num_buckets(stk::topology::ELEM_RANK),
                                                                                    Kokkos::AUTO);


    Kokkos::View<AABB*, stk::ngp::MemSpace> boxes("Example::boxes",N);

    Kokkos::parallel_for(teamPolicy,
        KOKKOS_LAMBDA(const TeamHandleType& team)
        {
            const stk::mesh::NgpMesh::BucketType& bucket = ngpMesh.get_bucket(stk::topology::ELEM_RANK,
                                                                              team.league_rank());
            unsigned numElems = bucket.size();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, numElems), [&] (const int& i)
            {
                stk::mesh::Entity particle = bucket[i];
                stk::mesh::FastMeshIndex idx = ngpMesh.fast_mesh_index(particle);

                ArborX::Point min_corner = {static_cast<float>(elemAabbField_device(idx, 0)), 
                                            static_cast<float>(elemAabbField_device(idx, 1)),
                                            static_cast<float>(elemAabbField_device(idx, 2))};

                ArborX::Point max_corner = {static_cast<float>(elemAabbField_device(idx, 3)), 
                                            static_cast<float>(elemAabbField_device(idx, 4)),
                                            static_cast<float>(elemAabbField_device(idx, 5))};

                ArborX::Box b = {min_corner, max_corner};
                boxes(i) = {b, particle};

            });
        }
    );

    stk::mesh::NgpMesh::MeshExecSpace execution_space{};

    ArborX::BVH<stk::ngp::MemSpace> index(execution_space, boxes);
    Kokkos::View<int *, stk::ngp::MemSpace> indices("Example::indices", 0);
    Kokkos::View<int *, stk::ngp::MemSpace> offsets("Example::offsets", 0);
    index.query(execution_space, boxes, ExcludeSelfCollision{}, indices, offsets);
    execution_space.fence();

    std::cout<<offsets.extent(0)<<endl;
    std::cout<<indices.extent(0)<<endl;
    
    neighbors.resize_serial(indices.extent(0));



    Kokkos::parallel_for("convert_search_res", N, KOKKOS_LAMBDA(int i){
        for (int j = offsets(i); j < offsets(i + 1); ++j)
        {
            int other = indices(j);
            neighbors(j) = Kokkos::make_pair(boxes(i).entity,boxes(j).entity);
        }
    });
}