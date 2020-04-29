#include <celero/Celero.h>

#include "Fixtures.hpp"

CELERO_MAIN

constexpr const int numberOfSamples = 1;
constexpr const int numberOfIterationsPerSample = 1;

template<typename ModelFiles>
using PolytopeSpace = PolytopeSpaceFixture<ModelFiles, Eigen::MatrixXd, Eigen::VectorXd>;

// TODO benchmark sparse vs dense csmala
BASELINE(reade_coli_core, CHR, numberOfSamples, numberOfIterationsPerSample) {
    PolytopeSpace<e_coli_core_files> polytopeSpaceFixture;
}
