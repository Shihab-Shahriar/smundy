#ifndef ELEMENT_H
#define ELEMENT_H

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <ArborX.hpp>
#include <ArborX_Version.hpp>

#include "Quaternion.hpp"
#include "smath.hpp"

constexpr double epsilon = Kokkos::Experimental::epsilon_v<double> * 100;


class Element{
public:
    uint64_t id;
    double coords[3], velocity[3], omega[3], force[3], torque[3];  //use Vec<3> instead. maybe replace with Eigen GPU? 
    ArborX::Box aabb;
};

class Sphere: public Element{
public:
    double orientation[4];
    double radius;
};

class Linker: public Element{
public:
    uint64_t ids[2];
    double signed_sep_dist;
    double signed_sep_dist_dot;
    double signed_sep_dist_dot_tmp;
    double lagrange_multiplier;
    double lagrange_multiplier_tmp;
    double constraint_attachment_locs[6];
    double constraint_attachment_norms[6];
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
    p.ids[0] = 0;
    p.ids[1] = 0;
}

template <typename T>
KOKKOS_FUNCTION ArborX::Box get_aabb(T& t);

template <>
KOKKOS_FUNCTION ArborX::Box get_aabb<Sphere>(Sphere& s){
    return s.aabb;    
}


//test this reducer
double compute_maximum_abs_projected_sep(Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers,
                                        double dt,
                                        double &global_maximum_abs_projected_sep){

    double local_maximum_abs_projected_sep = -1.0;
    int N = linkers.extent(0);
    Kokkos::parallel_reduce("swap_con_gammas",N, KOKKOS_LAMBDA(const int& i, double& sep){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        const double sep_new = linker.signed_sep_dist + dt * linker.signed_sep_dist_dot;

        double abs_projected_sep;
        if (linker.lagrange_multiplier < epsilon) {
            abs_projected_sep = Kokkos::abs(Kokkos::min(sep_new, 0.0));
        } else {
            abs_projected_sep = Kokkos::abs(sep_new);
        }

        if(abs_projected_sep > sep){
            sep = abs_projected_sep;
        }

    }, Kokkos::Max<double>(local_maximum_abs_projected_sep));
    global_maximum_abs_projected_sep = local_maximum_abs_projected_sep; //no global op on GPU
    return local_maximum_abs_projected_sep;
}

void compute_diff_dots(
    Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers,
    const double dt,
    double &global_dot_xkdiff_xkdiff,
    double &global_dot_xkdiff_gkdiff,
    double &global_dot_gkdiff_gkdiff)
{
    // compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff)
    // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1
    double local_dot_xkdiff_xkdiff = 0.0;
    double local_dot_xkdiff_gkdiff = 0.0;
    double local_dot_gkdiff_gkdiff = 0.0;

    int N = linkers.extent(0);
    Kokkos::parallel_reduce("swap_con_gammas",N, KOKKOS_LAMBDA(const int& i, double& xk_xk, double& xk_gk, double& gk_gk){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        const double xkdiff = linker.lagrange_multiplier - linker.lagrange_multiplier_tmp;
        const double gkdiff = dt * (linker.signed_sep_dist_dot - linker.signed_sep_dist_dot_tmp);

        xk_xk += xkdiff * xkdiff;
        xk_gk += xkdiff * gkdiff;
        gk_gk += gkdiff * gkdiff;

    }, local_dot_xkdiff_xkdiff, local_dot_xkdiff_gkdiff, local_dot_gkdiff_gkdiff); 

    global_dot_xkdiff_xkdiff = local_dot_xkdiff_xkdiff;
    global_dot_xkdiff_gkdiff = local_dot_xkdiff_gkdiff;
    global_dot_gkdiff_gkdiff = local_dot_gkdiff_gkdiff;

}

void generate_collision_constraints(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres,
                                    Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers, 
                                    Kokkos::View<int *, Kokkos::CudaSpace>& indices,
                                    Kokkos::View<int *, Kokkos::CudaSpace>& offsets)
{
    int N = offsets.extent(0)-1;
    Kokkos::RangePolicy<Kokkos::Cuda> rpolicy(Kokkos::Cuda{}, 0, N);
    Kokkos::parallel_for("generate_const",rpolicy, KOKKOS_LAMBDA(int i){
        Sphere& si = spheres(i);
        for(int j=offsets(i);j<offsets(i+1);j++){
            if(i>indices(j)) continue; //Don't generate two linkers for each pair

            Linker& linker = linkers(j);
            Sphere& sj = spheres(indices(j));

            Vec<double, 3> distIJ({sj.coords[0] - si.coords[0], sj.coords[1] - si.coords[1], sj.coords[2] - si.coords[2]});
            const double com_sep = Kokkos::sqrt(distIJ[0] * distIJ[0] + distIJ[1] * distIJ[1] + distIJ[2] * distIJ[2]);
            const Vec<double, 3> normIJ({distIJ[0]/com_sep,distIJ[1]/com_sep,distIJ[2]/com_sep });

            linker.signed_sep_dist = com_sep - si.radius - sj.radius;
            linker.signed_sep_dist_dot = 0.0;
            linker.signed_sep_dist_dot_tmp = 0.0;
            linker.lagrange_multiplier = 0.0;
            linker.lagrange_multiplier_tmp = 0.0;

            linker.constraint_attachment_locs[0] = normIJ[0] * si.radius;
            linker.constraint_attachment_locs[1] = normIJ[1] * si.radius;
            linker.constraint_attachment_locs[2] = normIJ[2] * si.radius;
            linker.constraint_attachment_locs[3] = -normIJ[0] * si.radius;
            linker.constraint_attachment_locs[4] = -normIJ[1] * si.radius;
            linker.constraint_attachment_locs[5] = -normIJ[2] * si.radius;

            linker.constraint_attachment_norms[0] = normIJ[0];
            linker.constraint_attachment_norms[1] = normIJ[1];
            linker.constraint_attachment_norms[2] = normIJ[2];
            linker.constraint_attachment_norms[3] = -normIJ[0];
            linker.constraint_attachment_norms[4] = -normIJ[1];
            linker.constraint_attachment_norms[5] = -normIJ[2];

            linker.ids[0] = i;
            linker.ids[1] = indices(j);
        }
    }
    );                            
                                
}

void compute_center_of_mass_force_and_torque(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres,
                                    Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers)
{

    // loop over spheres
    Kokkos::parallel_for("zero_out_force_&_torque",spheres.extent(0), KOKKOS_LAMBDA(int i){
        Sphere& si = spheres(i);
        si.force[0] = 0.0;
        si.force[1] = 0.0;
        si.force[2] = 0.0;

        si.torque[0] = 0.0;
        si.torque[1] = 0.0;
        si.torque[2] = 0.0;
    });

    // loop over linkers.
    int N = linkers.extent(0);
    Kokkos::RangePolicy<Kokkos::Cuda> rpolicy(Kokkos::Cuda{}, 0, N);
    Kokkos::parallel_for("comp_force_&_torque",rpolicy, KOKKOS_LAMBDA(int i){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        Sphere& si = spheres(linker.ids[0]);
        Sphere& sj = spheres(linker.ids[1]);


        const double linker_lag_mult = linker.lagrange_multiplier;

        si.force[0] += -linker_lag_mult * linker.constraint_attachment_norms[0];
        si.force[1] += -linker_lag_mult * linker.constraint_attachment_norms[1];
        si.force[2] += -linker_lag_mult * linker.constraint_attachment_norms[2];

        si.torque[0] += -linker_lag_mult * (linker.constraint_attachment_locs[1] * linker.constraint_attachment_norms[2] - linker.constraint_attachment_locs[2] * linker.constraint_attachment_norms[1]);
        si.torque[1] += -linker_lag_mult * (linker.constraint_attachment_locs[2] * linker.constraint_attachment_norms[0] - linker.constraint_attachment_locs[0] * linker.constraint_attachment_norms[2]);
        si.torque[2] += -linker_lag_mult * (linker.constraint_attachment_locs[0] * linker.constraint_attachment_norms[1] - linker.constraint_attachment_locs[1] * linker.constraint_attachment_norms[0]);


        sj.force[0] += -linker_lag_mult * linker.constraint_attachment_norms[3];
        sj.force[1] += -linker_lag_mult * linker.constraint_attachment_norms[4];
        sj.force[2] += -linker_lag_mult * linker.constraint_attachment_norms[5];

        sj.torque[0] += -linker_lag_mult * (linker.constraint_attachment_locs[4] * linker.constraint_attachment_norms[5] - linker.constraint_attachment_locs[5] * linker.constraint_attachment_norms[4]);
        sj.torque[1] += -linker_lag_mult * (linker.constraint_attachment_locs[5] * linker.constraint_attachment_norms[3] - linker.constraint_attachment_locs[3] * linker.constraint_attachment_norms[5]);
        sj.torque[2] += -linker_lag_mult * (linker.constraint_attachment_locs[3] * linker.constraint_attachment_norms[4] - linker.constraint_attachment_locs[4] * linker.constraint_attachment_norms[3]);

    });
}

// compute the mobility matrix for the sphere
void compute_the_mobility_problem(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres, double viscosity){

    Kokkos::parallel_for("compute_the_mobility_problem",spheres.extent(0), KOKKOS_LAMBDA(int i){
        Sphere& si = spheres(i);
        Quaternion quat(si.orientation[0],si.orientation[1],si.orientation[2],si.orientation[3] );
        Vec<double, 3> q = quat.rotate(Vec<double, 3>{0,0,1});

        const double qq[3][3] = {{q[0] * q[0], q[0] * q[1], q[0] * q[2]}, {q[1] * q[0], q[1] * q[1], q[1] * q[2]},
            {q[2] * q[0], q[2] * q[1], q[2] * q[2]}};
        const double Imqq[3][3] = {{1 - qq[0][0], -qq[0][1], -qq[0][2]}, {-qq[1][0], 1 - qq[1][1], -qq[1][2]},
            {-qq[2][0], -qq[2][1], 1 - qq[2][2]}};

        constexpr auto PI = Kokkos::numbers::pi_v<double>;
        const double drag_para = 6 * PI * si.radius * viscosity;
        const double drag_perp = drag_para;
        const double drag_rot = 8 * PI * si.radius * si.radius * si.radius * viscosity;
        const double drag_para_inv = 1.0 / drag_para;
        const double drag_perp_inv = 1.0 / drag_perp;
        const double drag_rot_inv = 1.0 / drag_rot;

        const double mob_trans[3][3] = {
          {drag_para_inv * qq[0][0] + drag_perp_inv * Imqq[0][0], drag_para_inv * qq[0][1] + drag_perp_inv * Imqq[0][1],
              drag_para_inv * qq[0][2] + drag_perp_inv * Imqq[0][2]},
          {drag_para_inv * qq[1][0] + drag_perp_inv * Imqq[1][0], drag_para_inv * qq[1][1] + drag_perp_inv * Imqq[1][1],
              drag_para_inv * qq[1][2] + drag_perp_inv * Imqq[1][2]},
          {drag_para_inv * qq[2][0] + drag_perp_inv * Imqq[2][0], drag_para_inv * qq[2][1] + drag_perp_inv * Imqq[2][1],
              drag_para_inv * qq[2][2] + drag_perp_inv * Imqq[2][2]}};
        const double mob_rot[3][3] = {
          {drag_rot_inv * qq[0][0] + drag_rot_inv * Imqq[0][0], drag_rot_inv * qq[0][1] + drag_rot_inv * Imqq[0][1],
              drag_rot_inv * qq[0][2] + drag_rot_inv * Imqq[0][2]},
          {drag_rot_inv * qq[1][0] + drag_rot_inv * Imqq[1][0], drag_rot_inv * qq[1][1] + drag_rot_inv * Imqq[1][1],
              drag_rot_inv * qq[1][2] + drag_rot_inv * Imqq[1][2]},
          {drag_rot_inv * qq[2][0] + drag_rot_inv * Imqq[2][0], drag_rot_inv * qq[2][1] + drag_rot_inv * Imqq[2][1],
              drag_rot_inv * qq[2][2] + drag_rot_inv * Imqq[2][2]}};

        si.velocity[0] =
          mob_trans[0][0] * si.force[0] + mob_trans[0][1] * si.force[1] + mob_trans[0][2] * si.force[2];
        si.velocity[1] =
            mob_trans[1][0] * si.force[0] + mob_trans[1][1] * si.force[1] + mob_trans[1][2] * si.force[2];
        si.velocity[2] =
            mob_trans[2][0] * si.force[0] + mob_trans[2][1] * si.force[1] + mob_trans[2][2] * si.force[2];

        si.omega[0] = mob_rot[0][0] * si.torque[0] + mob_rot[0][1] * si.torque[1] + mob_rot[0][2] * si.torque[2];
        si.omega[1] = mob_rot[1][0] * si.torque[0] + mob_rot[1][1] * si.torque[1] + mob_rot[1][2] * si.torque[2];
        si.omega[2] = mob_rot[2][0] * si.torque[0] + mob_rot[2][1] * si.torque[1] + mob_rot[2][2] * si.torque[2];

    });
}


void compute_rate_of_change_of_sep(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres,
                                    Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers)
{
    int N = linkers.extent(0);
    Kokkos::parallel_for("rate_of_change_of_sep",N, KOKKOS_LAMBDA(int i){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        Sphere& si = spheres(linker.ids[0]);
        Sphere& sj = spheres(linker.ids[1]);


        Vec<double, 3> com_velocityI(si.velocity);
        Vec<double, 3> com_velocityJ(sj.velocity);

        Vec<double, 3> com_omegaI(si.omega);
        Vec<double, 3> com_omegaJ(sj.omega);

        Vec<double, 3> con_posI{linker.constraint_attachment_locs[0],linker.constraint_attachment_locs[1],linker.constraint_attachment_locs[2]};
        Vec<double, 3> con_posJ{linker.constraint_attachment_locs[3],linker.constraint_attachment_locs[4],linker.constraint_attachment_locs[5]};

        Vec<double, 3> con_normI{linker.constraint_attachment_norms[0],linker.constraint_attachment_norms[1],linker.constraint_attachment_norms[2]};
        Vec<double, 3> con_normJ{linker.constraint_attachment_norms[3],linker.constraint_attachment_norms[4],linker.constraint_attachment_norms[5]};

        const Vec<double, 3> con_velI = com_velocityI + com_omegaI.cross(con_posI);
        const Vec<double, 3> con_velJ = com_velocityJ + com_omegaJ.cross(con_posJ);
        linker.signed_sep_dist_dot = -con_normI.dot(con_velI) - con_normJ.dot(con_velJ);
    });
}

void update_con_gammas(Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers, double alpha, double dt){
    int N = linkers.extent(0);
    Kokkos::parallel_for("update_con_gammas",N, KOKKOS_LAMBDA(int i){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        const double sep_new = linker.signed_sep_dist + dt* linker.signed_sep_dist_dot;
        double tmp = linker.lagrange_multiplier_tmp - alpha * sep_new;
        linker.lagrange_multiplier = tmp>0? tmp:0.0;
    });
}

void swap_con_gammas(Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers){
    int N = linkers.extent(0);
    Kokkos::parallel_for("swap_con_gammas",N, KOKKOS_LAMBDA(int i){
        Linker& linker = linkers(i);
        if(linker.ids[0]+linker.ids[1]==0) return; //one of the unused linkers

        linker.lagrange_multiplier_tmp = linker.lagrange_multiplier;
        linker.signed_sep_dist_dot_tmp = linker.signed_sep_dist_dot;
    });
}

void step_euler(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres, double dt){
    int N = spheres.extent(0);
    Kokkos::parallel_for("step_euler",N, KOKKOS_LAMBDA(int i){
        Sphere& si = spheres(i);

        si.coords[0] += dt * si.velocity[0];
        si.coords[1] += dt * si.velocity[1];
        si.coords[2] += dt * si.velocity[2];

        Quaternion quat(si.orientation[0],si.orientation[1],si.orientation[2],si.orientation[3]);
        quat.rotate_self(si.omega[0], si.omega[1], si.omega[2], dt);

        si.orientation[0] = quat.w;
        si.orientation[1] = quat.x;
        si.orientation[2] = quat.y;
        si.orientation[3] = quat.z;
    });
}

void resolve_collision(Kokkos::View<Sphere *, Kokkos::CudaSpace>& spheres,
                Kokkos::View<Linker *, Kokkos::CudaSpace>& linkers,
                const double viscosity,
                const double dt,
                const double con_tol,
                const int con_ite_max)
{
    // Matrix-free BBPGD
    int ite_count = 0;

    // compute gkm1 = D^T M D xkm1

    // compute F = D xkm1
    compute_center_of_mass_force_and_torque(spheres, linkers);

    // compute U = M F
    compute_the_mobility_problem(spheres, viscosity);

    // compute gkm1 = D^T U
    compute_rate_of_change_of_sep(spheres, linkers);


    double maximum_abs_projected_sep = -1.0;
    compute_maximum_abs_projected_sep(linkers, dt, maximum_abs_projected_sep);

    std::cout << "maximum_abs_projected_sep " << maximum_abs_projected_sep << std::endl;
 
    if (maximum_abs_projected_sep < con_tol) {
        // the initial guess was correct, nothing more is necessary
    }
    else{
        double alpha = 1.0 / maximum_abs_projected_sep;
        while (ite_count < con_ite_max) {
            ++ite_count;

            // compute xk = xkm1 - alpha * gkm1;
            // and perform the bound projection xk = boundProjection(xk)
            update_con_gammas(linkers, alpha, dt);

            // compute new grad with xk: gk = D^T M D xk
            // compute F = D xk
            compute_center_of_mass_force_and_torque(spheres, linkers);

            // compute U = M F
            compute_the_mobility_problem(spheres, viscosity);

            // compute gk = D^T U
            compute_rate_of_change_of_sep(spheres, linkers);

            compute_maximum_abs_projected_sep(linkers, dt, maximum_abs_projected_sep);

            std::cout << ite_count<<": maximum_abs_projected_sep " << maximum_abs_projected_sep << std::endl;

            if (maximum_abs_projected_sep < con_tol) {
                // con_gammas worked
                // exit the loop
                break;
            }

            ///////////////////////////////////////////////////////////////////////////
            // compute dot(xkdiff, xkdiff), dot(xkdiff, gkdiff), dot(gkdiff, gkdiff) //
            // where xkdiff = xk - xkm1 and gkdiff = gk - gkm1                       //
            ///////////////////////////////////////////////////////////////////////////
            double global_dot_xkdiff_xkdiff = 0.0;
            double global_dot_xkdiff_gkdiff = 0.0;
            double global_dot_gkdiff_gkdiff = 0.0;
            compute_diff_dots(linkers, dt, global_dot_xkdiff_xkdiff, global_dot_xkdiff_gkdiff, global_dot_gkdiff_gkdiff);


            ////////////////////////////////////////////
            // compute the Barzilai-Borwein step size //
            ////////////////////////////////////////////
            // alternating bb1 and bb2 methods
            double a;
            double b;
            if (ite_count % 2 == 0) {
                // Barzilai-Borwein step size Choice 1
                a = global_dot_xkdiff_xkdiff;
                b = global_dot_xkdiff_gkdiff;
            } else {
                // Barzilai-Borwein step size Choice 2
                a = global_dot_xkdiff_gkdiff;
                b = global_dot_gkdiff_gkdiff;
            }  

            if (std::abs(b) < epsilon) {
                b += epsilon;  // prevent div 0 error
            }
            alpha = a / b;

            swap_con_gammas(linkers);
        }
    }

}

#endif