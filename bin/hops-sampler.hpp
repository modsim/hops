#ifndef HOPS_SAMPLERS_HPP
#define HOPS_SAMPLERS_HPP

#include <hops/hops.hpp>

std::vector<Eigen::VectorXd> sampleUniformly(const Eigen::MatrixXd &A,
                                             const Eigen::VectorXd &b,
                                             const Eigen::VectorXd &startingPoint,
                                             const Eigen::MatrixXd &transformation,
                                             const Eigen::VectorXd &shift,
                                             long numberOfSamples,
                                             long thinning,
                                             int randomSeed) {

    auto markovChain = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::StateTransformation(
                            hops::CoordinateHitAndRunProposal(
                                    A,
                                    b,
                                    startingPoint),
                            hops::LinearTransformation(transformation, shift))
            )
    );

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);

    std::vector<Eigen::VectorXd> samples;
    for (long i = 0; i < numberOfSamples; ++i) {
        samples.emplace_back(markovChain.draw(randomNumberGenerator, thinning).second);
    }
    return samples;
}

std::vector<Eigen::VectorXd> sampleUniformly(const Eigen::MatrixXd &A,
                                             const Eigen::VectorXd &b,
                                             const Eigen::VectorXd &startingPoint,
                                             long numberOfSamples,
                                             long thinning,
                                             int randomSeed) {

    auto markovChain = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::CoordinateHitAndRunProposal(
                            A,
                            b,
                            startingPoint)
            )
    );

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);

    std::vector<Eigen::VectorXd> samples;
    for (long i = 0; i < numberOfSamples; ++i) {
        samples.emplace_back(markovChain.draw(randomNumberGenerator, thinning).second);
    }
    return samples;
}

#endif //HOPS_SAMPLERS_HPP
