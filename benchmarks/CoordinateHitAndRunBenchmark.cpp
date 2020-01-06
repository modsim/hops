#include <celero/Celero.h>
#include "Fixtures.hpp"

CELERO_MAIN

constexpr const int numberOfSamples = 1;
constexpr const int numberOfIterationsPerSample = 1000;

using Model = Recon2;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Polytope = PolytopeSpaceFixture<Model, Matrix, Vector>;

template<nups::MarkovChainType markovChainType>
using MarkovChain = MarkovChainFixture<Polytope, markovChainType>;

BASELINE_F(CoordinateHitAndRun, CHRRS,
           MarkovChain<nups::MarkovChainType::CoordinateHitAndRun>, numberOfSamples,
           numberOfIterationsPerSample) {
    markovChain->draw(randomNumberGenerator, 1);
}
