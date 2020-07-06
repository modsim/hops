#ifndef HOPS_GUROBIENVIRONMENTSINGLETON_HPP
#define HOPS_GUROBIENVIRONMENTSINGLETON_HPP

#include <gurobi_c++.h>

namespace hops {
    class GurobiEnvironmentSingleton {
        using GurobiEnvironment = GRBEnv;
    public:
        static GurobiEnvironmentSingleton &getInstance();

        const GurobiEnvironment &getGurobiEnvironment();

    private:
        GurobiEnvironmentSingleton();

        GurobiEnvironment environment;
    };
}

#endif //HOPS_GUROBIENVIRONMENTSINGLETON_HPP
