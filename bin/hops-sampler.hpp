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
                    hops::StateRecorder(
                            hops::StateTransformation(
                                    hops::CoordinateHitAndRunProposal(
                                            A,
                                            b,
                                            startingPoint),
                                    hops::Transformation(transformation, shift))
                    )
            )
    );

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);
    markovChain.draw(randomNumberGenerator, numberOfSamples, thinning);

    return markovChain.getStateRecords();
}

std::vector<Eigen::VectorXd> sampleUniformly(const Eigen::MatrixXd &A,
                                             const Eigen::VectorXd &b,
                                             const Eigen::VectorXd &startingPoint,
                                             long numberOfSamples,
                                             long thinning,
                                             int randomSeed) {

    auto markovChain = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::StateRecorder(
                            hops::CoordinateHitAndRunProposal(
                                    A,
                                    b,
                                    startingPoint)
                    )
            )
    );

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);
    markovChain.draw(randomNumberGenerator, numberOfSamples, thinning);

    return markovChain.getStateRecords();
}

#endif //HOPS_SAMPLERS_HPP
