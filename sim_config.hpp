#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include <cstddef>
#include <cmath>
#include <stk_mesh/base/Ngp.hpp>

struct Configuration{
    Configuration(){}
    const double viscosity = 0.001;
    const double dt = 5e-3;
    const double time_snap = 5e-3;
    const double time_stop = 1;

    const size_t num_elements_per_group = 640000;
    const double R = 0.133;
    const double cutoff = 2 * R;
    const double con_tol = 1e-5;
    const int con_ite_max = 1000;
    const unsigned int spatial_dimension = 3;
    const double domain_low[3] = {0.0, 0.0, 0.0};
    const double domain_high[3] = {40.0, 40.0, 40.0};
    const int NSpheres = num_elements_per_group;
    const int NLinkers = num_elements_per_group;
    const int Total_elements = NSpheres + NLinkers;
};

using EntityPair = Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>;
using SearchResultView = Kokkos::Experimental::DynamicView<EntityPair*, stk::ngp::MemSpace>;
using ConnectedEntities = stk::util::StridedArray<const stk::mesh::Entity>;
typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;
typedef stk::search::IdentProc<stk::mesh::EntityKey> SearchIdentProc;
typedef std::vector<std::pair<SearchIdentProc, SearchIdentProc> > SearchIdPairVector;
typedef std::vector<std::pair<stk::search::Box<double>, SearchIdentProc> > BoxIdVector;


#endif