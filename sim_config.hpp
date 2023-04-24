#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include <cstddef>
#include <stk_mesh/base/Ngp.hpp>

struct Configuration{
    Configuration(){}
    const double viscosity = 0.001;
    const double dt = 5e-3;
    const double time_snap = 5e-3;
    const double time_stop = 1;

    const double R = 4.0f;
    const double cutoff = 2 * R;
    const double con_tol = 1e-5;
    const int con_ite_max = 1000;
    const unsigned int spatial_dimension = 3;
    const double domain_low[3] = {0.0, 0.0, 0.0};
    const double domain_high[3] = {30.0, 30.0, 30.0};
    const size_t num_elements_per_group = 270000;
    const int NSpheres = num_elements_per_group;
    const int NLinkers = num_elements_per_group;
    const int Total_elements = NSpheres + NLinkers;
};

using EntityPair = Kokkos::pair<stk::mesh::Entity, stk::mesh::Entity>;
typedef stk::ngp::TeamPolicy<stk::mesh::NgpMesh::MeshExecSpace>::member_type TeamHandleType;


#endif