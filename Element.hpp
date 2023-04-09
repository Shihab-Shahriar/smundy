#ifndef ELEMENT_H
#define ELEMENT_H

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <ArborX.hpp>
#include <ArborX_Version.hpp>

#include "Quaternion.hpp"

class Element{
public:
    uint64_t id;
    double coords[3], velocity[3], omega[3], force[3], torque[3];
    ArborX::Box aabb;
};

class Sphere: public Element{
public:
    double orientation[4];
    double radius;
};

class Linker: public Element{
    double signed_sep_dist;
    double signed_sep_dist_dot;
    double signed_sep_dist_dot_tmp;
    double lagrange_multiplier;
    double lagrange_multiplier_tmp;
    double constraint_attachment_locs;
    double constraint_attachment_norms;
};

template <typename T, typename U>
KOKKOS_FUNCTION void random_init(T& t, U& gen, const Configuration& CONFIG){

}

using RANDOM_TYPE = Kokkos::Random_XorShift64<Kokkos::CudaSpace>;

template<>
KOKKOS_FUNCTION void random_init<Element, RANDOM_TYPE>(Element& e, RANDOM_TYPE& gen, const Configuration& CONFIG){

    e.coords[0] = gen.drand(0.0, 1.0) * (CONFIG.domain_high[0] - CONFIG.domain_low[0]) + CONFIG.domain_low[0];
    e.coords[1] = gen.drand(0.0, 1.0) * (CONFIG.domain_high[1] - CONFIG.domain_low[1]) + CONFIG.domain_low[1];
    e.coords[2] = gen.drand(0.0, 1.0) * (CONFIG.domain_high[2] - CONFIG.domain_low[2]) + CONFIG.domain_low[2];

    float R = static_cast<float>(CONFIG.R);

    ArborX::Point min_corner = {static_cast<float>(e.coords[0]) - R,static_cast<float>(e.coords[1]) - R,static_cast<float>(e.coords[2]) - R};
    ArborX::Point max_corner = {static_cast<float>(e.coords[0]) + R,static_cast<float>(e.coords[1]) + R,static_cast<float>(e.coords[2]) + R};

    e.aabb = {min_corner, max_corner};

    e.omega[0] = 0.0;
    e.omega[1] = 0.0;
    e.omega[2] = 0.0;

    e.velocity[0] = 0.0;
    e.velocity[1] = 0.0;
    e.velocity[2] = 0.0;

    e.force[0] = 0.0;
    e.force[1] = 0.0;
    e.force[2] = 0.0;

    e.torque[0] = 0.0;
    e.torque[1] = 0.0;
    e.torque[2] = 0.0;
}

template<>
KOKKOS_FUNCTION void random_init<Sphere, RANDOM_TYPE> (Sphere& p, RANDOM_TYPE& gen, const Configuration& CONFIG){
    random_init<Element,RANDOM_TYPE> (p, gen, CONFIG);
    p.radius = CONFIG.R;

    const double u1 = gen.drand(0.0, 1.0);
    const double u2 = gen.drand(0.0, 1.0);
    const double u3 = gen.drand(0.0, 1.0);

    Quaternion quat(u1, u2, u3);
    p.orientation[0] = quat.w;
    p.orientation[1] = quat.x;
    p.orientation[2] = quat.y;
    p.orientation[3] = quat.z;
}

template<>
KOKKOS_FUNCTION void random_init<Linker, RANDOM_TYPE> (Linker& p, RANDOM_TYPE& gen, const Configuration& CONFIG){
    random_init<Element,RANDOM_TYPE> (p, gen, CONFIG);
}

template <typename T>
KOKKOS_FUNCTION ArborX::Box get_aabb(T& t);

template <>
KOKKOS_FUNCTION ArborX::Box get_aabb<Sphere>(Sphere& s){
    return s.aabb;    
}

#endif

