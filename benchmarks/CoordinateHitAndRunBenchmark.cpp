#include <celero/Celero.h>
#include "Fixtures.hpp"

CELERO_MAIN

constexpr const int numberOfSamples = 1;
constexpr const int numberOfIterationsPerSample = 10;

using Model = Recon2;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Polytope = PolytopeSpaceFixture<Model, Matrix, Vector>;

template<hops::MarkovChainType markovChainType>
using MarkovChain = MarkovChainFixture<Polytope, markovChainType>;

BASELINE_F(CoordinateHitAndRun, CHRRS,
           MarkovChain<hops::MarkovChainType::CoordinateHitAndRun>, numberOfSamples,
           numberOfIterationsPerSample) {
    markovChain->draw(randomNumberGenerator, 10, 242800);
}
