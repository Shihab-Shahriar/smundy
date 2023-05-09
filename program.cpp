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
using namespace mundy;

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    Kokkos::ScopeGuard guard(argc, argv);

    const Configuration CONFIG;
    const int spatial_dimension = CONFIG.spatial_dimension;

    stk::topology particle_top = stk::topology::PARTICLE;
    stk::topology link_top = stk::topology::BEAM_2;

    stk::mesh::MeshBuilder builder(MPI_COMM_WORLD);
    builder.set_spatial_dimension(spatial_dimension);
    builder.set_entity_rank_names({"node", "edge", "face", "elem"});

    std::shared_ptr<stk::mesh::BulkData> bulkPtr = builder.create();
    bulkPtr->mesh_meta_data().use_simple_fields();

    // declare the parts
    stk::mesh::MetaData &metaData = bulkPtr->mesh_meta_data();
    stk::mesh::Part &linkerPart = metaData.declare_part_with_topology("linePart", link_top);
    stk::mesh::Part &particlePart = metaData.declare_part_with_topology("particlePart", particle_top);
    stk::io::put_io_part_attribute(linkerPart);
    stk::io::put_io_part_attribute(particlePart);

    // declare and assign fields
    // node fields (shared between particles and linkers)
    stk::mesh::Field<double> &nodeCoordField =
        metaData.declare_field<double>(stk::topology::NODE_RANK, "coordinates", spatial_dimension);
    stk::mesh::Field<double> &nodeVelocityField =
        metaData.declare_field<double>(stk::topology::NODE_RANK, "velocity", spatial_dimension);
    stk::mesh::Field<double> &nodeOmegaField =
        metaData.declare_field<double>(stk::topology::NODE_RANK, "omega", spatial_dimension);
    stk::mesh::Field<double> &nodeForceField =
        metaData.declare_field<double>(stk::topology::NODE_RANK, "force", spatial_dimension);
    stk::mesh::Field<double> &nodeTorqueField =
        metaData.declare_field<double>(stk::topology::NODE_RANK, "torque", spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(nodeCoordField, spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(nodeVelocityField, spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(nodeOmegaField, spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(nodeForceField, spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(nodeTorqueField, spatial_dimension);

    // element fields (stored on both particles and linkers)
    stk::mesh::Field<int> &elemRankField = metaData.declare_field<int>(stk::topology::ELEMENT_RANK, "rank");
    // assign a unique id to each local element
    stk::mesh::Field<int> &elemLocalIDField = metaData.declare_field<int>(stk::topology::ELEMENT_RANK, "local_id");

    stk::mesh::Field<double> &elemAabbField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "aabb", 2 * spatial_dimension);
    stk::mesh::put_field_on_entire_mesh(elemRankField);
    stk::mesh::put_field_on_entire_mesh(elemLocalIDField);
    stk::mesh::put_field_on_entire_mesh(elemAabbField, 2 * spatial_dimension);

    // element fields (stored only on particles)
    stk::mesh::Field<double> &particleOrientationField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "orientation", spatial_dimension + 1);
    stk::mesh::Field<double> &particleRadiusField = metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "radius");
    stk::mesh::put_field_on_mesh(particleOrientationField, particlePart, spatial_dimension + 1, nullptr);
    stk::mesh::put_field_on_mesh(particleRadiusField, particlePart, nullptr);

    // element fields (stored only on linkers)
    stk::mesh::Field<double> &linkerSignedSepField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist");
    stk::mesh::Field<double> &linkerSignedSepDotField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist_dot");
    stk::mesh::Field<double> &linkerSignedSepDotTmpField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "signed_sep_dist_dot_tmp");
    stk::mesh::Field<double> &linkerLagMultField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "lagrange_multiplier");
    stk::mesh::Field<double> &linkerLagMultTmpField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "lagrange_multiplier_tmp");
    stk::mesh::Field<double> &conLocField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "constraint_attachment_locs", 2 * spatial_dimension);
    stk::mesh::Field<double> &conNormField =
        metaData.declare_field<double>(stk::topology::ELEMENT_RANK, "constraint_attachment_norms", 2 * spatial_dimension);
    stk::mesh::put_field_on_mesh(linkerSignedSepField, linkerPart, nullptr);
    stk::mesh::put_field_on_mesh(linkerSignedSepDotField, linkerPart, nullptr);
    stk::mesh::put_field_on_mesh(linkerSignedSepDotTmpField, linkerPart, nullptr);
    stk::mesh::put_field_on_mesh(linkerLagMultField, linkerPart, nullptr);
    stk::mesh::put_field_on_mesh(linkerLagMultTmpField, linkerPart, nullptr);
    stk::mesh::put_field_on_mesh(conLocField, linkerPart, 2 * spatial_dimension, nullptr);
    stk::mesh::put_field_on_mesh(conNormField, linkerPart, 2 * spatial_dimension, nullptr);

    metaData.set_coordinate_field_name("coordinates");
    metaData.commit();

    // construct the mesh in parallel
    stk::mesh::BulkData &bulkData = *bulkPtr;

    // get the averge number of particles per process
    const int num_particles_global = CONFIG.num_elements_per_group;
    size_t num_particles_local = num_particles_global / bulkData.parallel_size();
    printf("N = %d, dim = %f\n", num_particles_local, CONFIG.domain_high[0]);

    // num_particles_local isn't guarenteed to divide perfectly
    // add the extra workload to the first r ranks
    size_t remaining_particles = num_particles_global - num_particles_local * bulkData.parallel_size();
    if (bulkData.parallel_rank() < remaining_particles) {
        num_particles_local += 1;
    }

    Kokkos::Timer timer;
    bulkData.modification_begin();

    std::vector<size_t> requests(metaData.entity_rank_count(), 0);
    const size_t num_nodes_requested = num_particles_local * particle_top.num_nodes();
    const size_t num_elems_requested = num_particles_local;
    requests[stk::topology::NODE_RANK] = num_nodes_requested;
    requests[stk::topology::ELEMENT_RANK] = num_elems_requested;

    // ex.
    //  requests = { 0, 4,  8}
    //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
    //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
    std::vector<stk::mesh::Entity> requested_entities;
    bulkData.generate_new_entities(requests, requested_entities);

    // associate each particle with a single part
    std::vector<stk::mesh::Part *> add_particlePart(1);
    add_particlePart[0] = &particlePart;

    // set topologies of new entities
    for (int i = 0; i < num_particles_local; i++) {
        stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
        bulkData.change_entity_parts(particle_i, add_particlePart);
    }

    // the elements should be associated with a topology before they are connected to their nodes/edges
    // set downward relations of entities
    for (int i = 0; i < num_particles_local; i++) {
        stk::mesh::Entity particle_i = requested_entities[num_nodes_requested + i];
        bulkData.declare_relation(particle_i, requested_entities[i], 0);
    }
    bulkData.modification_end();

    cout<<"Started checking mod_end"<<endl;
for(stk::mesh::EntityRank rank=stk::topology::EDGE_RANK; rank<=stk::topology::ELEMENT_RANK; ++rank) {
    cout<<rank<<endl;
    const stk::mesh::BucketVector& buckets = bulkData.buckets(rank);
    for(size_t i=0; i<buckets.size(); ++i) {
      const stk::mesh::Bucket& bucket = *buckets[i];
      if (bucket.topology() == stk::topology::INVALID_TOPOLOGY && bucket.size() > 0)
      {
        std::cerr << "Entities on rank " << rank << " bucket " << i << " have no topology defined" << std::endl;
        return -1;
      }
      for(size_t j=0; j<bucket.size(); ++j) {
        if (bucket.num_nodes(j) < 1) {
          std::cerr << "Entity with rank="<<rank<<", identifier="<<bulkData.identifier(bucket[j])<<" has no connected nodes."<<std::endl;
          return -1;
        }
        // NEED TO CHECK FOR EACH BUCKET INHABITANT THAT ALL ITS NODES ARE VALID.
        unsigned num_nodes = bucket.num_nodes(j);
        stk::mesh::Entity const* nodes = bucket.begin_nodes(j);
        for (unsigned k = 0; k < num_nodes; ++k) {
          if (!bulkData.is_valid(nodes[k])) {
            std::cerr << "Entity with rank="<<rank<<", identifier="<<bulkData.identifier(bucket[j])<<" is connected to an invalid node."
                      << " via node relation " << k << std::endl;
            return -1;
          }
        }
      }
    }
  }

    double time_to_mod = timer.seconds();

    cout<<"Hey:"<<num_particles_local<<endl;
    //printf("%d %d %d \n", num_particles_local, bulkData.parallel_size(), bulkData.parallel_rank());
    cout<<time_to_mod<<endl;
    cout<<endl;

    //init GPU stuff
    Kokkos::Random_XorShift64_Pool<stk::ngp::MemSpace> random_pool(12345);
    // Common Element Fields
    stk::mesh::NgpField<int>& elemRankField_device = stk::mesh::get_updated_ngp_field<int>(elemRankField);
    stk::mesh::NgpField<int>& elemLocalIDField_device = stk::mesh::get_updated_ngp_field<int>(elemLocalIDField);
    stk::mesh::NgpField<double>& particleOrientationField_device = stk::mesh::get_updated_ngp_field<double>(particleOrientationField);
    stk::mesh::NgpField<double>& particleRadiusField_device = stk::mesh::get_updated_ngp_field<double>(particleRadiusField);
    stk::mesh::NgpField<double>& elemAabbField_device = stk::mesh::get_updated_ngp_field<double>(elemAabbField);

    // Linker GPU fields
    stk::mesh::NgpField<double>& linkerSignedSepField_device = stk::mesh::get_updated_ngp_field<double>(linkerSignedSepField);
    stk::mesh::NgpField<double>& linkerSignedSepDotField_device = stk::mesh::get_updated_ngp_field<double>(linkerSignedSepDotField);
    stk::mesh::NgpField<double>& linkerSignedSepDotTmpField_device = stk::mesh::get_updated_ngp_field<double>(linkerSignedSepDotTmpField);
    stk::mesh::NgpField<double>& linkerLagMultField_device = stk::mesh::get_updated_ngp_field<double>(linkerLagMultField);
    stk::mesh::NgpField<double>& linkerLagMultTmpField_device = stk::mesh::get_updated_ngp_field<double>(linkerLagMultTmpField);
    stk::mesh::NgpField<double>& conLocField_device = stk::mesh::get_updated_ngp_field<double>(conLocField);
    stk::mesh::NgpField<double>& conNormField_device = stk::mesh::get_updated_ngp_field<double>(conNormField);


    //Node fields
    stk::mesh::NgpField<double>& nodeCoordField_device = stk::mesh::get_updated_ngp_field<double>(nodeCoordField);
    stk::mesh::NgpField<double>& nodeVelocityField_device = stk::mesh::get_updated_ngp_field<double>(nodeVelocityField);
    stk::mesh::NgpField<double>& nodeOmegaField_device = stk::mesh::get_updated_ngp_field<double>(nodeOmegaField);
    stk::mesh::NgpField<double>& nodeForceField_device = stk::mesh::get_updated_ngp_field<double>(nodeForceField);
    stk::mesh::NgpField<double>& nodeTorqueField_device = stk::mesh::get_updated_ngp_field<double>(nodeTorqueField);
    stk::mesh::NgpMesh & ngpMesh = stk::mesh::get_updated_ngp_mesh(bulkData);
    cout<<"begin init..."<<endl;



    //Initialize
    {
    elemAabbField_device.modify_on_device();
    elemRankField_device.modify_on_device();
    particleRadiusField_device.modify_on_device();
    particleOrientationField_device.modify_on_device();
    nodeCoordField_device.modify_on_device();
    nodeOmegaField_device.modify_on_device();
    nodeVelocityField_device.modify_on_device();
    nodeForceField_device.modify_on_device();
    nodeTorqueField_device.modify_on_device();


    const auto& teamPolicy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(ngpMesh.num_buckets(stk::topology::ELEM_RANK),
                                                                                    Kokkos::AUTO);

    
    timer.reset();
    Kokkos::parallel_for(teamPolicy,
        KOKKOS_LAMBDA(const TeamHandleType& team)
        {
            const stk::mesh::NgpMesh::BucketType& bucket = ngpMesh.get_bucket(stk::topology::ELEM_RANK,
            team.league_rank());
            unsigned numElems = bucket.size();

            // put all CONFIG vals in shared mem scope?
            const double R = CONFIG.R;


            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 0u, numElems), [&] (const int& i)
            {
                stk::mesh::Entity particle = bucket[i];
                stk::mesh::FastMeshIndex elemIndex = ngpMesh.fast_mesh_index(particle);

                Kokkos::Random_XorShift64<stk::ngp::MemSpace> gen = random_pool.get_state();

                elemRankField_device(elemIndex, 0) = 1;

                //on the assumption that max upto 512 elements per bucket
                elemLocalIDField_device(elemIndex, 0) = team.league_rank () * 512 + team.team_rank ();
                particleRadiusField_device(elemIndex, 0) = R;

                // random initial orientation
                const double u1 = gen.drand(0.0, 1.0);
                const double u2 = gen.drand(0.0, 1.0);
                const double u3 = gen.drand(0.0, 1.0);

                Quaternion quat(u1, u2, u3);
                particleOrientationField_device(elemIndex, 0) = quat.w;
                particleOrientationField_device(elemIndex, 1) = quat.x;
                particleOrientationField_device(elemIndex, 2) = quat.y;
                particleOrientationField_device(elemIndex, 3) = quat.z;

                stk::mesh::Entity node = ngpMesh.get_nodes(stk::topology::ELEM_RANK, elemIndex)[0];
                stk::mesh::FastMeshIndex nodeIndex = ngpMesh.fast_mesh_index(node);

                nodeCoordField_device(nodeIndex, 0) = gen.drand(0.0, 1.0) * (CONFIG.domain_high[0] - CONFIG.domain_low[0]) + CONFIG.domain_low[0];
                nodeCoordField_device(nodeIndex, 1) = gen.drand(0.0, 1.0) * (CONFIG.domain_high[1] - CONFIG.domain_low[1]) + CONFIG.domain_low[1];
                nodeCoordField_device(nodeIndex, 2) = gen.drand(0.0, 1.0) * (CONFIG.domain_high[2] - CONFIG.domain_low[2]) + CONFIG.domain_low[2];


                elemAabbField_device(elemIndex, 0) = nodeCoordField_device(nodeIndex, 0) - R;
                elemAabbField_device(elemIndex, 1) = nodeCoordField_device(nodeIndex, 1) - R;
                elemAabbField_device(elemIndex, 2) = nodeCoordField_device(nodeIndex, 2) - R;
                elemAabbField_device(elemIndex, 3) = nodeCoordField_device(nodeIndex, 0) + R;
                elemAabbField_device(elemIndex, 4) = nodeCoordField_device(nodeIndex, 1) + R;
                elemAabbField_device(elemIndex, 5) = nodeCoordField_device(nodeIndex, 2) + R;


                nodeOmegaField_device(nodeIndex, 0) = 0.0;
                nodeOmegaField_device(nodeIndex, 1) = 0.0;
                nodeOmegaField_device(nodeIndex, 2) = 0.0;

                nodeVelocityField_device(nodeIndex, 0) = 0.0;
                nodeVelocityField_device(nodeIndex, 1) = 0.0;
                nodeVelocityField_device(nodeIndex, 2) = 0.0;

                nodeForceField_device(nodeIndex, 0) = 0.0;
                nodeForceField_device(nodeIndex, 1) = 0.0;
                nodeForceField_device(nodeIndex, 2) = 0.0;

                nodeTorqueField_device(nodeIndex, 0) = 0.0;
                nodeTorqueField_device(nodeIndex, 1) = 0.0;
                nodeTorqueField_device(nodeIndex, 2) = 0.0;

                random_pool.free_state(gen);

            });
        }
    );
    Kokkos::fence();
    double time1 = timer.seconds();
    cout<<time1<<endl;
    }
    cout<<"Initted"<<endl;

    timer.reset();
    
    //const int N = CONFIG.num_elements_per_group;

    // Figure out optimal chunk_size value for the constructor below.
    // Kokkos::Experimental::DynamicView<EntityPair*, stk::ngp::MemSpace> neighbors("Search_pairs", 16*1024, N*N);
    // I couldn't make DynamicView work- the host version using (index) always results in segfault. 

    cout<<"modification_end: "<<elemAabbField_device.need_sync_to_host()<<" "<<elemAabbField_device.need_sync_to_device()<<endl;

    // modification_begin() synchronizes all device fields to host.
    bulkData.modification_begin();
    bulkData.modification_end();
    //elemAabbField_device.sync_to_host();
    cout<<"modification_end: "<<elemAabbField_device.need_sync_to_host()<<" "<<elemAabbField_device.need_sync_to_device()<<endl;
    
    timer.reset();
    SearchIdPairVector neighbors;
    generate_neighbor_pairs(bulkData, elemAabbField, neighbors);
    cout<<"neighbors generated: "<<neighbors.size()<<endl;
    double time2 = timer.seconds();
    cout<<"generate_neighbor_pairs:"<<time2<<endl;

    timer.reset();
    generate_collision_constraints(bulkData, neighbors, linkerPart, nodeCoordField, particleRadiusField,
      linkerSignedSepField, linkerSignedSepDotField, linkerSignedSepDotTmpField, linkerLagMultField,
      linkerLagMultTmpField, conLocField, conNormField);
    nodeCoordField.modify_on_host();
    nodeCoordField.sync_to_device();
    particleRadiusField.modify_on_host();
    particleRadiusField.sync_to_device();
    linkerSignedSepField.modify_on_host();
    linkerSignedSepField.sync_to_device();
    linkerSignedSepDotField.modify_on_host();
    linkerSignedSepDotField.sync_to_device();
    linkerSignedSepDotTmpField.modify_on_host();
    linkerSignedSepDotTmpField.sync_to_device();
    linkerLagMultField.modify_on_host();
    linkerLagMultField.sync_to_device();
    linkerLagMultTmpField.modify_on_host();
    linkerLagMultTmpField.sync_to_device();
    conLocField.modify_on_host();
    conLocField.sync_to_device();
    conNormField.modify_on_host();
    conNormField.sync_to_device();

    time2 = timer.seconds();
    cout<<"generate_collision_constraints:"<<time2<<endl;

    
    ngpMesh.update_mesh(); //sync from host to device.
    //cout<<"Mesh updated? "<<updated<<endl;

    double time_start  = stk::wall_time();
    resolve_collision(ngpMesh, nodeCoordField_device,  nodeForceField_device, nodeTorqueField_device,nodeVelocityField_device,
        nodeOmegaField_device, linkerLagMultField_device, linkerLagMultTmpField_device,
        linkerSignedSepField_device, linkerSignedSepDotField_device, linkerSignedSepDotTmpField_device,
        conNormField_device, conLocField_device, particleOrientationField_device, particleRadiusField_device, 
        CONFIG.viscosity, CONFIG.dt, CONFIG.con_tol, CONFIG.con_ite_max);
    double time_end  = stk::wall_time();
    double elapsed = time_end - time_start;
    cout<<"resolve_collision: "<<elapsed<<endl;

    MPI_Finalize();
    return 0;
}

