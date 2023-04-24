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

void compute_maximum_abs_projected_sep(const stk::mesh::NgpMesh & ngpMesh,
                                         stk::mesh::NgpField<double>& linkerLagMultField_device,
                                         stk::mesh::NgpField<double>& linkerSignedSepField_device,
                                         stk::mesh::NgpField<double>& linkerSignedSepDotField_device,
                                         double dt,
                                         double &global_maximum_abs_projected_sep)
{
    // Only for local machine

    // could instead use for reduction: 
    // get_field_reduction(Mesh &mesh, Field field, const stk::mesh::Selector &selector, ReductionOp& reduction, Modifier fm, 
    //                     const int & component = -1)

    double local_maximum_abs_projected_sep = -1.0;
    const stk::mesh::MetaData &metaData = ngpMesh.get_bulk_on_host().mesh_meta_data();
    stk::mesh::Selector selectLocalLinkers = 
        metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
    stk::NgpVector<unsigned> linkerBuckets = ngpMesh.get_bucket_ids(stk::topology::ELEMENT_RANK, selectLocalLinkers);

    const auto& teamPolicy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(linkerBuckets.size(),
                                                                                    Kokkos::AUTO);


    Kokkos::parallel_reduce(teamPolicy,
        KOKKOS_LAMBDA(const TeamHandleType& team, double& local_sep)
        {
            const stk::mesh::NgpMesh::BucketType& bucket = ngpMesh.get_bucket(stk::topology::ELEM_RANK,
                team.league_rank());
            unsigned numElems = bucket.size();
            double team_maximum_abs_projected_sep = -1.0;

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, 0u, numElems), [&] (const int& i, double& sep)
            {
                stk::mesh::Entity linker = bucket[i];
                stk::mesh::FastMeshIndex linkerIdx = ngpMesh.fast_mesh_index(linker);

                const double lag_mult = linkerLagMultField_device(linkerIdx, 0);
                const double sep_old = linkerSignedSepField_device(linkerIdx, 0);
                const double sep_dot = linkerSignedSepDotField_device(linkerIdx, 0);
                const double sep_new = sep_old + dt * sep_dot;

                double abs_projected_sep;
                if (lag_mult < epsilon) {
                    abs_projected_sep = Kokkos::abs(Kokkos::min(sep_new, 0.0));
                } else {
                    abs_projected_sep = Kokkos::abs(sep_new);
                }
                if(abs_projected_sep > sep){
                    sep = abs_projected_sep;
                }
            }, Kokkos::Max<double>(team_maximum_abs_projected_sep));

            // Only one thread should do this. Doesn't matter for min/max reductions though.
            if(team_maximum_abs_projected_sep> local_sep){
                local_sep = team_maximum_abs_projected_sep;
            }

        }, Kokkos::Max<double>(local_maximum_abs_projected_sep)
    );
    global_maximum_abs_projected_sep = local_maximum_abs_projected_sep; //no global op on GPU
}


// compute_diff_dots: Outer reducer can work on multiple variables at once, inner one can't. 
struct Diff_vals{
    KOKKOS_INLINE_FUNCTION Diff_vals(){
        team_dot_xkdiff_xkdiff = 0.0;
        team_dot_xkdiff_gkdiff = 0.0;
        team_dot_gkdiff_gkdiff = 0.0;
    }
    KOKKOS_INLINE_FUNCTION Diff_vals(const Diff_vals& rhs){
        team_dot_xkdiff_xkdiff = rhs.team_dot_xkdiff_xkdiff;
        team_dot_xkdiff_gkdiff = rhs.team_dot_xkdiff_gkdiff;
        team_dot_gkdiff_gkdiff = rhs.team_dot_gkdiff_gkdiff;
    }
    KOKKOS_INLINE_FUNCTION Diff_vals& operator += (const Diff_vals& rhs) {
        team_dot_xkdiff_xkdiff += rhs.team_dot_xkdiff_xkdiff;
        team_dot_xkdiff_gkdiff += rhs.team_dot_xkdiff_gkdiff;
        team_dot_gkdiff_gkdiff += rhs.team_dot_gkdiff_gkdiff;
        return *this;
    }
    double team_dot_xkdiff_xkdiff = 0.0;
    double team_dot_xkdiff_gkdiff = 0.0;
    double team_dot_gkdiff_gkdiff = 0.0;
}; 

namespace Kokkos { //reduction identity must be defined in Kokkos namespace
    template<>
    struct reduction_identity< Diff_vals > {
        KOKKOS_FORCEINLINE_FUNCTION static Diff_vals sum() {
            return Diff_vals();
        }
    };
}


void compute_diff_dots(const stk::mesh::NgpMesh & ngpMesh,
    const stk::mesh::NgpField<double> &linkerLagMultField,
    const stk::mesh::NgpField<double> &linkerLagMultTmpField,
    const stk::mesh::NgpField<double> &linkerSignedSepDotField,
    const stk::mesh::NgpField<double> &linkerSignedSepDotTmpField,
    const double dt,
    double &global_dot_xkdiff_xkdiff,
    double &global_dot_xkdiff_gkdiff,
    double &global_dot_gkdiff_gkdiff)
{
    // TODO: Write a custom reducer to be able to use stk's get_field_reduction. Compare that with this implementatoon

    // compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff)
    // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1
    double local_dot_xkdiff_xkdiff = 0.0;
    double local_dot_xkdiff_gkdiff = 0.0;
    double local_dot_gkdiff_gkdiff = 0.0;


    const stk::mesh::MetaData &metaData = ngpMesh.get_bulk_on_host().mesh_meta_data();
    stk::mesh::Selector selectLocalLinkers = 
        metaData.locally_owned_part() & metaData.get_topology_root_part(stk::topology::BEAM_2);
    stk::NgpVector<unsigned> linkerBuckets = ngpMesh.get_bucket_ids(stk::topology::ELEMENT_RANK, selectLocalLinkers);

    const auto& teamPolicy = stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>(linkerBuckets.size(),
                                                                                    Kokkos::AUTO);
    Kokkos::parallel_reduce(
        teamPolicy,
        KOKKOS_LAMBDA(const TeamHandleType& team, double& xk_xk, double& xk_gk, double& gk_gk){
            const stk::mesh::NgpMesh::BucketType& bucket = ngpMesh.get_bucket(stk::topology::ELEM_RANK,
                team.league_rank());
            unsigned numElems = bucket.size();
            

            Diff_vals team_val;

            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, 0u, numElems), 
                [&](const int& i, Diff_vals& vals){ 
                    stk::mesh::Entity linker = bucket[i];
                    stk::mesh::FastMeshIndex linkerIdx = ngpMesh.fast_mesh_index(linker);

                    // fetch the fields
                    const double lag_mult = linkerLagMultField(linkerIdx, 0);
                    const double lag_mult_tmp = linkerLagMultTmpField(linkerIdx, 0);
                    const double sep_dot = linkerSignedSepDotField(linkerIdx, 0); 
                    const double sep_dot_tmp = linkerSignedSepDotTmpField(linkerIdx, 0);

                    // xkdiff = xk - xkm1
                    const double xkdiff = lag_mult - lag_mult_tmp;

                    // gkdiff = gk - gkm1
                    const double gkdiff = dt * (sep_dot - sep_dot_tmp);

                    vals.team_dot_xkdiff_xkdiff += xkdiff * xkdiff;
                    vals.team_dot_xkdiff_gkdiff += xkdiff * gkdiff;
                    vals.team_dot_gkdiff_gkdiff += gkdiff * gkdiff;

                },
                Kokkos::Sum<Diff_vals>(team_val)
            ); 

            Kokkos::single(Kokkos::PerTeam(team), [&](){
                xk_xk += team_val.team_dot_xkdiff_xkdiff;
                xk_gk += team_val.team_dot_xkdiff_gkdiff;
                gk_gk += team_val.team_dot_gkdiff_gkdiff;
            });
        },
        local_dot_xkdiff_xkdiff, local_dot_xkdiff_gkdiff, local_dot_gkdiff_gkdiff
    );
    // don't care that about other ranks on GPU code.
    global_dot_xkdiff_xkdiff = local_dot_xkdiff_xkdiff;
    global_dot_xkdiff_gkdiff = local_dot_xkdiff_gkdiff;
    global_dot_gkdiff_gkdiff = local_dot_gkdiff_gkdiff;
}

// Copy pasted from UnitTestMundy
void create_ghosting(stk::mesh::BulkData &bulkData, const SearchResultView& searchResults, const std::string &name)
{
  ThrowRequire(bulkData.in_modifiable_state());
  const int parallel_rank = bulkData.parallel_rank();
  std::vector<stk::mesh::EntityProc> send_nodes;
  for (size_t i = 0; i < searchResults.size(); ++i) {
    stk::mesh::Entity domain_node = searchResults[i].first;
    stk::mesh::Entity range_node = searchResults[i].second;

    bool is_owned_domain = bulkData.is_valid(domain_node) ? bulkData.bucket(domain_node).owned() : false;
    bool is_owned_range = bulkData.is_valid(range_node) ? bulkData.bucket(range_node).owned() : false;
    int domain_proc = searchResults[i].first.proc();
    int range_proc = searchResults[i].second.proc();

    if (is_owned_domain && domain_proc == parallel_rank) {
      if (range_proc == parallel_rank) continue;

      ThrowRequire(bulkData.parallel_owner_rank(domain_node) == domain_proc);

      send_nodes.emplace_back(domain_node, range_proc);
    } else if (is_owned_range && range_proc == parallel_rank) {
      if (domain_proc == parallel_rank) continue;

      ThrowRequire(bulkData.parallel_owner_rank(range_node) == range_proc);

      send_nodes.emplace_back(range_node, domain_proc);
    }
  }

  stk::mesh::Ghosting &ghosting = bulkData.create_ghosting(name);
  bulkData.change_ghosting(ghosting, send_nodes);
}

// Copy pasted from UnitTestMundy
void generate_collision_constraints(stk::mesh::BulkData &bulkData,
    const SearchResultView& neighborPairs,
    stk::mesh::Part &linkerPart,
    stk::mesh::Field<double> &nodeCoordField,
    stk::mesh::Field<double> &particleRadiusField,
    stk::mesh::Field<double> &linkerSignedSepField,
    stk::mesh::Field<double> &linkerSignedSepDotField,
    stk::mesh::Field<double> &linkerSignedSepDotTmpField,
    stk::mesh::Field<double> &linkerLagMultField,
    stk::mesh::Field<double> &linkerLagMultTmpField,
    stk::mesh::Field<double> &conLocField,
    stk::mesh::Field<double> &conNormField)
{
  /*
  Note:
    A niave procedure can generate two linkers between every pair of particles
    This can be remedied by having the particle with the smaller ID generate the constraints

  Procedure:
    1. ghost neighbors that aren't on the current proocess
    2. generate linkers between neighbor particles
      2.1. the process that owns the particle with the smaller ID generates the linker
      2.2. add node sharing between processors
    3. fill the linkers with the collision information
  */

  // populate the ghost using the search results
  bulkData.modification_begin();
  create_ghosting(bulkData, neighborPairs, "geometricGhosts");
  bulkData.modification_end();

  // communicate the necessary ghost particle fields
  std::vector<const stk::mesh::FieldBase*> fields{&nodeCoordField, &particleRadiusField};
  stk::mesh::communicate_field_data(bulkData, fields);

  // generate linkers between the neighbors
  // at this point, the number of neighbors == the number of linkers that need generated
  // the particles already have nodes, so we only need to generate linker entities
  // and declare relations/sharing between those entities and the connected nodes
  bulkData.modification_begin();
  const size_t num_linkers = std::count_if(
      neighborPairs.begin(), neighborPairs.end(), [](const std::pair<SearchIdentProc, SearchIdentProc> &neighborPair) {
        return neighborPair.first.id() < neighborPair.second.id();
      });
  std::vector<size_t> requests(bulkData.mesh_meta_data().entity_rank_count(), 0);
  requests[stk::topology::ELEMENT_RANK] = num_linkers;

  // ex.
  //  requests = { 0, 4,  8}
  //  requests 0 entites of rank 0, 4 entites of rank 1, and 8 entites of rank 2
  //  requested_entities = {0 entites of rank 0, 4 entites of rank 1, 8 entites of rank 2}
  std::vector<stk::mesh::Entity> requested_entities;
  bulkData.generate_new_entities(requests, requested_entities);

  // associate each particle with a single part
  std::vector<stk::mesh::Part *> add_linkerPart(1);
  add_linkerPart[0] = &linkerPart;

  // set topologies of new entities
  // #pragma omp parallel for
  for (int i = 0; i < num_linkers; i++) {
    stk::mesh::Entity linker_i = requested_entities[i];
    bulkData.change_entity_parts(linker_i, add_linkerPart);
  }

  // the elements should be associated with a topology before they are connected to their nodes/edges
  // set downward relations of entities
  // loop over the neighbor pairs
  // #pragma omp parallel for
  size_t count = 0;
  for (int i = 0; i < neighborPairs.size(); i++) {
    stk::mesh::Entity particleI = bulkData.get_entity(neighborPairs[i].first.id());
    stk::mesh::Entity particleJ = bulkData.get_entity(neighborPairs[i].second.id());
    stk::mesh::Entity nodesI = bulkData.begin_nodes(particleI)[0];
    stk::mesh::Entity nodesJ = bulkData.begin_nodes(particleJ)[0];
    int owningProcI = neighborPairs[i].first.proc();
    int owningProcJ = neighborPairs[i].second.proc();

    // share both nodes with other process
    // add_node_sharing must be called symmetrically
    EXPECT_TRUE(owningProcI == bulkData.parallel_rank());
    if (bulkData.parallel_rank() != owningProcJ) {
      bulkData.add_node_sharing(nodesI, owningProcJ);
      bulkData.add_node_sharing(nodesJ, owningProcJ);
    }

    // only generate linkers if the source particle's id is less
    // than the id of the target particle. this prevents duplicate constraints
    if (neighborPairs[i].first.id() < neighborPairs[i].second.id()) {
      stk::mesh::Entity linker_i = requested_entities[count];
      bulkData.declare_relation(linker_i, nodesI, 0);
      bulkData.declare_relation(linker_i, nodesJ, 1);

      // fill the constraint information
      const double *const posI = stk::mesh::field_data(nodeCoordField, nodesI);
      const double *const posJ = stk::mesh::field_data(nodeCoordField, nodesJ);
      const double *const radiusI = stk::mesh::field_data(particleRadiusField, particleI);
      const double *const radiusJ = stk::mesh::field_data(particleRadiusField, particleJ);

      const stk::math::Vec<double, 3> distIJ({posJ[0] - posI[0], posJ[1] - posI[1], posJ[2] - posI[2]});
      const double com_sep = sqrt(distIJ[0] * distIJ[0] + distIJ[1] * distIJ[1] + distIJ[2] * distIJ[2]);
      const stk::math::Vec<double, 3> normIJ = distIJ / com_sep;

      stk::mesh::field_data(linkerSignedSepField, linker_i)[0] = com_sep - radiusI[0] - radiusJ[0];
      stk::mesh::field_data(linkerSignedSepDotField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerSignedSepDotTmpField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerLagMultField, linker_i)[0] = 0.0;
      stk::mesh::field_data(linkerLagMultTmpField, linker_i)[0] = 0.0;

      // con loc is relative to the com TODO: change name to be more explicit
      stk::mesh::field_data(conLocField, linker_i)[0] = radiusI[0] * normIJ[0];
      stk::mesh::field_data(conLocField, linker_i)[1] = radiusI[0] * normIJ[1];
      stk::mesh::field_data(conLocField, linker_i)[2] = radiusI[0] * normIJ[2];
      stk::mesh::field_data(conLocField, linker_i)[3] = -radiusJ[0] * normIJ[0];
      stk::mesh::field_data(conLocField, linker_i)[4] = -radiusJ[0] * normIJ[1];
      stk::mesh::field_data(conLocField, linker_i)[5] = -radiusJ[0] * normIJ[2];

      stk::mesh::field_data(conNormField, linker_i)[0] = normIJ[0];
      stk::mesh::field_data(conNormField, linker_i)[1] = normIJ[1];
      stk::mesh::field_data(conNormField, linker_i)[2] = normIJ[2];
      stk::mesh::field_data(conNormField, linker_i)[3] = -normIJ[0];
      stk::mesh::field_data(conNormField, linker_i)[4] = -normIJ[1];
      stk::mesh::field_data(conNormField, linker_i)[5] = -normIJ[2];

      count++;
    }
  }
  bulkData.modification_end();
}


void resolve_collision(){
        
    double sep = -1.0;
    compute_maximum_abs_projected_sep(ngpMesh, linkerLagMultField_device, linkerSignedSepField_device, 
                                      linkerSignedSepDotField_device, CONFIG.dt, sep);
    double global_dot_xkdiff_xkdiff = 0.0;
    double global_dot_xkdiff_gkdiff = 0.0;
    double global_dot_gkdiff_gkdiff = 0.0;
    compute_diff_dots(  ngpMesh, linkerLagMultField_device, linkerLagMultTmpField_device, 
                        linkerSignedSepDotField_device, linkerSignedSepDotTmpField_device,
                        CONFIG.dt, global_dot_xkdiff_xkdiff, global_dot_xkdiff_gkdiff, global_dot_gkdiff_gkdiff);
}

