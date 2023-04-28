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


void generate_neighbor_pairs__(const stk::mesh::NgpMesh & ngpMesh,
                              const stk::mesh::BulkData &bulkData, 
                              stk::mesh::NgpField<double>& elemAabbField_device, 
                              SearchIdPairVector& neighbors,
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

    std::cout<<offsets.extent(0)<<std::endl;
    std::cout<<indices.extent(0)<<std::endl;
    
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto indices_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
    auto boxes_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, boxes);

    neighbors.reserve(indices_host.size());
    for(int i=0; i<N; i++){
      if(i==100) std::cout<<i<<" "<<neighbors.size()<<std::endl;
      if(i==200) std::cout<<i<<" "<<neighbors.size()<<std::endl;
      if(i==500) std::cout<<i<<" "<<neighbors.size()<<std::endl;
      if(i%1000==0) std::cout<<i<<" "<<neighbors.size()<<std::endl;
        if(i%1000==0) std::cout<<i<<" "<<neighbors.size()<<std::endl;
        for (int j = offsets_host(i); j < offsets_host(i + 1); ++j)
        {
          auto left_idp = stk::search::IdentProc(bulkData.entity_key(boxes_host(i).entity), bulkData.parallel_rank());
          auto right_idp = stk::search::IdentProc(bulkData.entity_key(boxes_host(indices_host(j)).entity), bulkData.parallel_rank());

          neighbors.push_back(std::make_pair(left_idp, right_idp));
        }
    }
    assert(neighbors.size()==indices_host.size());

}




void filterOutSelfOverlap(const stk::mesh::BulkData &bulkData, SearchIdPairVector &searchResults)
{
  size_t numFiltered = 0;

  for (const auto &searchResult : searchResults) {
    stk::mesh::Entity element1 = bulkData.get_entity(searchResult.first.id());
    stk::mesh::Entity element2 = bulkData.get_entity(searchResult.second.id());
    int owningProcElement1 = searchResult.first.proc();
    int owningProcElement2 = searchResult.second.proc();

    ThrowRequireWithSierraHelpMsg(
        owningProcElement1 == bulkData.parallel_rank() || owningProcElement2 == bulkData.parallel_rank());

    bool anyIntersections = false;

    if (element1 == element2) {
      anyIntersections = true;
    }

    if (!anyIntersections) {
      searchResults[numFiltered] = searchResult;
      numFiltered++;
    }
  }

  searchResults.resize(numFiltered);
}

void filterOutNonLocalResults(const stk::mesh::BulkData &bulkData, SearchIdPairVector &searchResults)
{
  const int rank = bulkData.parallel_rank();
  size_t numFiltered = 0;

  for (const auto &searchResult : searchResults) {
    if (searchResult.first.proc() == rank) {
      searchResults[numFiltered] = searchResult;
      numFiltered++;
    }
  }

  searchResults.resize(numFiltered);
}

void generate_neighbor_pairs(const stk::mesh::BulkData &bulkData,
    const stk::mesh::Field<double> &elemAabbField,
    SearchIdPairVector &neighborPairs)
{
  // setup the search boxes (for each element)
  const stk::mesh::MetaData &metaData = bulkData.mesh_meta_data();
  BoxIdVector elementBoxes;

  const int rank = bulkData.parallel_rank();
  const size_t num_local_elements =
      stk::mesh::count_entities(bulkData, stk::topology::ELEM_RANK, metaData.locally_owned_part());
  elementBoxes.reserve(num_local_elements);

  const stk::mesh::BucketVector &elementBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, metaData.locally_owned_part());
  for (size_t bucket_idx = 0; bucket_idx < elementBuckets.size(); ++bucket_idx) {
    stk::mesh::Bucket &elemBucket = *elementBuckets[bucket_idx];
    for (size_t elem_idx = 0; elem_idx < elemBucket.size(); ++elem_idx) {
      stk::mesh::Entity const &element = elemBucket[elem_idx];

      double *aabb = stk::mesh::field_data(elemAabbField, element);
      stk::search::Box<double> box(aabb[0], aabb[1], aabb[2], aabb[3], aabb[4], aabb[5]);

      SearchIdentProc search_id(bulkData.entity_key(element), rank);

      elementBoxes.emplace_back(box, search_id);
    }
  }

  // perform the aabb search
  std::cout<<"about to search...."<<std::endl;
  stk::search::coarse_search(elementBoxes, elementBoxes, stk::search::KDTREE, bulkData.parallel(), neighborPairs);

  // filter results
  // remove self-overlap and remove nonlocal source boxes
  filterOutSelfOverlap(bulkData, neighborPairs);
  filterOutNonLocalResults(bulkData, neighborPairs);
}