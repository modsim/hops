#ifdef HOPS_GUROBI_FOUND

#include "GurobiEnvironmentSingleton.hpp"

hops::GurobiEnvironmentSingleton &hops::GurobiEnvironmentSingleton::getInstance() {
    static GurobiEnvironmentSingleton instance;
    return instance;
}

const hops::GurobiEnvironmentSingleton::GurobiEnvironment &hops::GurobiEnvironmentSingleton::getGurobiEnvironment() {
    return environment;
}

hops::GurobiEnvironmentSingleton::GurobiEnvironmentSingleton() : environment(true) {
    environment.set(GRB_IntParam_LogToConsole, 0);
    environment.start();
    environment.set(GRB_IntParam_LogToConsole, 0);
    environment.set(GRB_IntParam_ScaleFlag, 2);
    environment.set(GRB_IntParam_NumericFocus, 3);
    environment.set(GRB_IntParam_Quad, 1);
    environment.set(GRB_DoubleParam_FeasibilityTol, 1e-9);
    environment.set(GRB_DoubleParam_OptimalityTol, 1e-9);
    environment.set(GRB_DoubleParam_MarkowitzTol, 0.999);
}

#endif //HOPS_GUROBI_FOUND
