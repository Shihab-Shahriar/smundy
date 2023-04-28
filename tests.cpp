#include <vector>                                    // for vector, etc
#include "mpi.h"                                     // for MPI_COMM_WORLD, etc
#include "omp.h"                                     // for pargma omp parallel, etc
#include <iostream>
#include <fstream>

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


void output_field_on_file(){
    std::ofstream ofs;
    ofs.open("aabb.txt");
    const stk::mesh::BucketVector &elementBuckets =
      bulkData.get_buckets(stk::topology::ELEMENT_RANK, metaData.locally_owned_part());
    for (size_t bucket_idx = 0; bucket_idx < elementBuckets.size(); ++bucket_idx) {
        stk::mesh::Bucket &elemBucket = *elementBuckets[bucket_idx];
        for (size_t elem_idx = 0; elem_idx < elemBucket.size(); ++elem_idx) {
            stk::mesh::Entity const &element = elemBucket[elem_idx];

            double *aabb = stk::mesh::field_data(elemAabbField, element);
            ofs<<aabb[0]<<","<< aabb[1]<<","<< aabb[2]<<","<< aabb[3]<<","<< aabb[4]<<","<< aabb[5]<<endl;
        }
    }
    ofs.close();
}