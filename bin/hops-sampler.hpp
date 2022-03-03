#ifndef HOPS_SAMPLERS_HPP
#define HOPS_SAMPLERS_HPP

#include <hops/hops.hpp>

std::tuple<std::vector<Eigen::VectorXd>, std::vector<long>, std::vector<long>> sampleUniformly(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const Eigen::VectorXd &startingPoint,
        const Eigen::MatrixXd &transformation,
        const Eigen::VectorXd &shift,
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


    auto transform = hops::LinearTransformation(transformation, shift);

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);

    std::vector<Eigen::VectorXd> samples;
    std::vector<long> sample_times;
    std::vector<long> transform_times;
    samples.reserve(numberOfSamples);

    for (long i = 0; i < numberOfSamples; ++i) {
        auto start_sample_time = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sample = markovChain.draw(randomNumberGenerator, thinning).second;
        auto end_sample_time_start_transform_time = std::chrono::high_resolution_clock::now();
        sample = transform.apply(sample);
        auto end_transform_time = std::chrono::high_resolution_clock::now();

        long sample_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_sample_time_start_transform_time - start_sample_time
        ).count();
        long transform_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_transform_time - end_sample_time_start_transform_time
        ).count();

        samples.emplace_back(sample);
        sample_times.emplace_back(sample_time);
        transform_times.emplace_back(transform_time);
    }
    return std::make_tuple(samples, sample_times, transform_times);
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
