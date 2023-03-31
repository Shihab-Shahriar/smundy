
#include <iostream>
#include <Kokkos_Core.hpp>

const int DIM = 3;

using namespace std;

class Particle{
public:
    double coordinates[3], velocity[3], omega[3], force[3], torque[3];
    double aabb[6];
    double orientation[4];
}

class Linker: Particle{
    double boo=99;
}

int main(){

    Linker b;

    cout<<b.aabb[0]<<endl;

    return 0;
}