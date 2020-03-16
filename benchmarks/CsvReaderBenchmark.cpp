#include <celero/Celero.h>

#include "Fixtures.hpp"

CELERO_MAIN

constexpr const int numberOfSamples = 1;
constexpr const int numberOfIterationsPerSample = 1;

template<typename ModelFiles>
using PolytopeSpace = PolytopeSpaceFixture<ModelFiles, Eigen::MatrixXd, Eigen::VectorXd>;

BASELINE(readRecon3D, CHR, numberOfSamples, numberOfIterationsPerSample) {
    PolytopeSpace<Recon3D> polytopeSpaceFixture;
}
