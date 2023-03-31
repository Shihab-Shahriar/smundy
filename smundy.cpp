#include <iostream>

#include <Kokkos_Core.hpp>

#include "Quaternion.hpp"
#include "smath.hpp"




int main(){
    const double viscosity = 0.001;
    const double dt = 5e-3;
    const double time_snap = 5e-3;
    const double time_stop = 1;

    const double R = 0.133;
    const double cutoff = 2 * R;
    const double con_tol = 1e-5;
    const int con_ite_max = 1000;
    const unsigned int spatial_dimension = 3;
    const double domain_low[3] = {0.0, 0.0, 0.0};
    const double domain_high[3] = {30.0, 30.0, 30.0};
    const size_t num_particles_global = 2700;

    

}