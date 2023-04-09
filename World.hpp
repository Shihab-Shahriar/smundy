#ifndef WORLD_H
#define WORLD_H


#include <vector>
#include <utility>
#include <algorithm>

#include "sim_config.hpp"

using namespace std;

class World{

public:
    void initialize(){

    }

    void get_neighbors(vector<pair<int, int>>& neighs){
        neighs.push_back(make_pair(20,19));
    }
};

#endif
