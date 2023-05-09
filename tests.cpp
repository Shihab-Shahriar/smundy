#include <vector>                                    // for vector, etc
#include "mpi.h"                                     // for MPI_COMM_WORLD, etc
#include "omp.h"                                     // for pargma omp parallel, etc
#include <iostream>
#include <fstream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_DynamicView.hpp>
#include <ArborX.hpp>
#include <ArborX_Version.hpp>


#include <type_traits>
#include <gtest/gtest.h>                             // for AssertHelper, etc
#include <stddef.h>                                  // for size_t
#include <stk_math/StkVector.hpp>                    // foor Vec
#include <stk_mesh/base/BulkData.hpp>                // for BulkData
#include <stk_mesh/base/MetaData.hpp>                // for MetaData
#include <stk_mesh/base/GetEntities.hpp>             // for count_selected_entities
#include <stk_mesh/base/Types.hpp>                   // for EntityVector, etc
#include <stk_mesh/base/Ghosting.hpp>                // for create_ghosting
#include <stk_io/WriteMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_search/SearchMethod.hpp>               // for KDTREE
#include <stk_search/CoarseSearch.hpp>               // for coarse_search
#include <stk_search/BoundingBox.hpp>                // for Sphere, Box, tec.
#include <stk_balance/balance.hpp>                   // for balanceStkMesh
#include <stk_util/parallel/Parallel.hpp>            // for ParallelMachine
#include <stk_util/environment/WallTime.hpp>         // for wall_time
#include <stk_util/environment/perf_util.hpp>
#include <stk_unit_test_utils/BuildMesh.hpp>
#include <stk_mesh/base/Ngp.hpp>
#include <stk_mesh/base/NgpAtomics.hpp>
#include <stk_mesh/base/NgpReductions.hpp>
#include <stk_mesh/base/NgpMesh.hpp>
#include <stk_mesh/base/GetNgpMesh.hpp>
#include <stk_mesh/base/NgpField.hpp>
#include <stk_mesh/base/GetNgpField.hpp>


#include "sim_config.hpp"
#include "Quaternion.hpp"
#include "smath.hpp"
#include "neighbor_search.hpp"
#include "collision_kernels.hpp"


using namespace std;

TEST(GPU, HelloWorld){
    int x = 1;
    EXPECT_EQ(x, 1) <<"trouble "<<endl;
    //EXPECT_EQ(x, 2) <<"supposed to fail "<<endl;

}

void run_vector_gpu_test()
{
  size_t n = 10;
  stk::NgpVector<double> vec("vec", n);
  Kokkos::parallel_for(stk::ngp::DeviceRangePolicy(0, n),
                       KOKKOS_LAMBDA(const int i)
                       {
                         vec.device_get(i) = i;
                       });
  vec.copy_device_to_host();
  for(size_t i=0; i<n; i++)
    EXPECT_EQ(i, vec[i]);
}

TEST(GPU, gpu_runs)
{
  Kokkos::ScopeGuard guard();
  run_vector_gpu_test();
}


// class UpdateNgpMesh : public stk::unit_test_util::simple_fields::MeshFixture
// {
// public:
//   void setup_test_mesh()
//   {
//     setup_empty_mesh(stk::mesh::BulkData::NO_AUTO_AURA);
//     std::string meshDesc = "0,1,HEX_8,1,2,3,4,5,6,7,8\n";
//     stk::unit_test_util::simple_fields::setup_text_mesh(get_bulk(), meshDesc);
//   }
// };

// TEST_F(UpdateNgpMesh, lazyAutoUpdate)
// {
//   setup_test_mesh();

//   // Don't store persistent pointers/references if you want automatic updates
//   // when acquiring an NgpMesh from BulkData
//   stk::mesh::NgpMesh * ngpMesh = &stk::mesh::get_updated_ngp_mesh(get_bulk());

//   get_bulk().modification_begin();
//   get_bulk().modification_end();

// #ifdef STK_USE_DEVICE_MESH
//   EXPECT_FALSE(ngpMesh->is_up_to_date());
//   ngpMesh = &stk::mesh::get_updated_ngp_mesh(get_bulk());
//   EXPECT_TRUE(ngpMesh->is_up_to_date());
// #else
//   EXPECT_TRUE(ngpMesh->is_up_to_date());
//   ngpMesh = &stk::mesh::get_updated_ngp_mesh(get_bulk());
//   EXPECT_TRUE(ngpMesh->is_up_to_date());
// #endif
// }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard guard(argc, argv);

    return RUN_ALL_TESTS();
}