#include <iostream>
#include <fstream>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>


#include "smath.hpp"
#include "Quaternion.hpp"

using namespace std;

using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;


int main(int argc, char* argv[]){
    ofstream myfile, kfile;
    myfile.open ("mat.csv");
    kfile.open ("kfile.csv");

    Kokkos::ScopeGuard kokkos(argc, argv);

    int N = 24000;
    Kokkos::View<double**> A("A", N, N);
    Kokkos::View<double*> x("x", N);

    Kokkos::Random_XorShift64_Pool<> rand_pool(42);
    typedef typename Kokkos::Random_XorShift64_Pool<>::generator_type gen_type;

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> md_policy({0,0},{N,N});
    Kokkos::parallel_for("init", md_policy, KOKKOS_LAMBDA(int i, int j){
        gen_type rgen = rand_pool.get_state();
        A(i,j) = Kokkos::rand<gen_type, double>::draw(rgen);
        if(i==0) x(j) = Kokkos::rand<gen_type, double>::draw(rgen);
        rand_pool.free_state(rgen);
    });

    auto A_h = Kokkos::create_mirror(A);
    auto x_h = Kokkos::create_mirror(x);
    Kokkos::deep_copy(A_h, A);
    Kokkos::deep_copy(x_h, x);
    Kokkos::fence();

    // for(int i=0;i<N;i++){
    //     myfile<<A_h(i,0);
    //     for(int j=1;j<N;j++){
    //         myfile<<","<<A_h(i,j);
    //     }
    //     myfile<<"\n";
    // }
    // myfile<<x_h(0);
    // for(int i=1;i<N;i++) myfile<<","<<x_h(i);
    myfile.close();

    Kokkos::Timer timer;
    timer.reset();
    Kokkos::View<double*> y_a("x", N);
    Kokkos::parallel_for("dot_a", Kokkos::RangePolicy<>(0,N), KOKKOS_LAMBDA(int row){
        double s = 0.0;
        for(int col=0;col<N;col++) s+= A(row, col)*x(col);
        y_a(row) = s;
    });
    Kokkos::fence();
    double time1 = timer.seconds();
    
    auto ya_h = Kokkos::create_mirror(y_a);
    Kokkos::deep_copy(ya_h, y_a);
    Kokkos::fence();

    kfile<<ya_h(0);
    for(int i=1;i<N;i++) kfile<<","<<ya_h(i);
    kfile<<"\n";


    timer.reset();
    Kokkos::TeamPolicy<> tpolicy(N, Kokkos::AUTO);
    typedef typename Kokkos::TeamPolicy<>::member_type member_type;

    Kokkos::View<double*> y_b("y_b", N);
    Kokkos::parallel_for("dot_b", tpolicy, KOKKOS_LAMBDA(member_type team){
        double s = 0.0;
        int row = team.league_rank();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, N), [&](const int col, double& sum){
            sum += A(row, col)*x(col);
        }, s);
        y_b(row) = s;
    });
    Kokkos::fence();
    double time2 = timer.seconds();

    auto yb_h = Kokkos::create_mirror(y_b);
    Kokkos::deep_copy(yb_h, y_b);
    Kokkos::fence();

    kfile<<yb_h(0);
    for(int i=1;i<N;i++) kfile<<","<<yb_h(i);
    kfile.close();

    cout<<time1<<" "<<time2<<endl;

    return 0;

}