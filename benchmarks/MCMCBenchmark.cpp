#include <celero/Celero.h>
#include "Fixtures.hpp"

CELERO_MAIN

constexpr const int numberOfSamples = 0;
constexpr const int numberOfIterationsPerSample = 0;

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Polytope = PolytopeSpaceFixture<iJO1366_files, Matrix, Vector>;

template<hops::MarkovChainType markovChainType>
using MarkovChain = MarkovChainFixture<Polytope, markovChainType>;

BASELINE_F(CSmMALA, CHRRS,
           MarkovChain<hops::MarkovChainType::CSmMALA>, numberOfSamples,
           numberOfIterationsPerSample) {
    markovChain->draw(randomNumberGenerator, 1, 1);
}
