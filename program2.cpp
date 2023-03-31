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
    ofstream myfile;
    myfile.open ("mat.csv");

    Kokkos::ScopeGuard kokkos(argc, argv);

    // Kokkos::DefaultExecutionSpace{}.print_configuration(std::cout, true);
    // cout<<"default....."<<endl;
    // Kokkos::DefaultHostExecutionSpace{}.print_configuration(std::cout, true);


    int N = 80000;
    Kokkos::View<Vec<double,4>*, Kokkos::DefaultExecutionSpace> a("a", N);

    Kokkos::Random_XorShift64_Pool<> rand_pool(42);
    typedef typename Kokkos::Random_XorShift64_Pool<>::generator_type gen_type;

    Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(int i){
        gen_type rgen = rand_pool.get_state();
        double x = Kokkos::rand<gen_type, double>::draw(rgen);
        double y = Kokkos::rand<gen_type, double>::draw(rgen);
        double z = Kokkos::rand<gen_type, double>::draw(rgen);
        double q = Kokkos::rand<gen_type, double>::draw(rgen);
        rand_pool.free_state(rgen);

        a(i) = a(i) + Vec<double,4>{x,y,z,q};
    });

    Kokkos::View<int*, Kokkos::DefaultExecutionSpace> flag("flag", 400);
    TeamPolicy<> policy(10, Kokkos::AUTO);
    typedef TeamPolicy<>::member_type member_type;

    Kokkos::parallel_for("add", policy, KOKKOS_LAMBDA(member_type team_member){
        int k = team_member.league_rank () * team_member.team_size () +
            team_member.team_rank ();
        flag(k) = 1;
    });


    auto x = Kokkos::create_mirror(flag);

    Kokkos::deep_copy( x, flag);

    for(int i=0;i<400;i++){
        if(x[i]) cout<<i<<endl;
    }

    myfile.close();
    return 0;

}