#ifndef HOPS_SAMPLERS_HPP
#define HOPS_SAMPLERS_HPP

#include "hops/hops.hpp"

std::tuple<hops::RandomNumberGenerator, std::unique_ptr<hops::MarkovChain>, std::unique_ptr<hops::Transformation>>
setUpSampling(
        const Eigen::MatrixXd &A,
        const Eigen::VectorXd &b,
        const Eigen::VectorXd &startingPoint,
        const Eigen::MatrixXd &transformation,
        const Eigen::VectorXd &shift,
        int randomSeed) {

    auto markovChainImpl = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::CoordinateHitAndRunProposal(
                            A,
                            b,
                            startingPoint)
            )
    );

    std::unique_ptr<hops::MarkovChain> markovChain =
            std::make_unique<decltype(markovChainImpl)>(markovChainImpl);


    std::unique_ptr<hops::Transformation> transform =
            std::make_unique<hops::LinearTransformation>(transformation, shift);

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);
    return std::make_tuple(randomNumberGenerator, std::move(markovChain), std::move(transform));
}


std::tuple<hops::RandomNumberGenerator, std::unique_ptr<hops::MarkovChain>>
setUpSampling(const Eigen::MatrixXd &A,
              const Eigen::VectorXd &b,
              const Eigen::VectorXd &startingPoint,
              int randomSeed) {

    auto markovChainImpl = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::CoordinateHitAndRunProposal(
                            A,
                            b,
                            startingPoint)
            )
    );

    std::unique_ptr<hops::MarkovChain> markovChain = std::make_unique<decltype(markovChainImpl)>(markovChainImpl);

    hops::RandomNumberGenerator randomNumberGenerator(randomSeed);

    return std::make_tuple(randomNumberGenerator, std::move(markovChain));
}


std::tuple<std::vector<Eigen::VectorXd>, std::vector<long>, std::vector<long>> sampleUniformBatch(
        long batch_size,
        long thinning,
        hops::RandomNumberGenerator &randomNumberGenerator,
        hops::MarkovChain *markovChain,
        hops::Transformation *transform
) {

    std::vector<Eigen::VectorXd> samples;
    std::vector<long> sample_times;
    std::vector<long> transform_times;
    samples.reserve(batch_size);

    for (long i = 0; i < batch_size; ++i) {
        auto start_sample_time = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sample = markovChain->draw(randomNumberGenerator, thinning).second;
        auto end_sample_time_start_transform_time = std::chrono::high_resolution_clock::now();
        sample = transform->apply(sample);
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


std::tuple<std::vector<Eigen::VectorXd>, std::vector<long>> sampleUniformBatch(
        long batch_size,
        long thinning,
        hops::RandomNumberGenerator &randomNumberGenerator,
        hops::MarkovChain *markovChain) {

    std::vector<Eigen::VectorXd> samples;
    std::vector<long> sample_times;
    samples.reserve(batch_size);

    for (long i = 0; i < batch_size; ++i) {
        auto start_sample_time = std::chrono::high_resolution_clock::now();
        Eigen::VectorXd sample = markovChain->draw(randomNumberGenerator, thinning).second;
        auto end_sample_time = std::chrono::high_resolution_clock::now();

        long sample_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end_sample_time - start_sample_time
        ).count();

        samples.emplace_back(sample);
        sample_times.emplace_back(sample_time);
    }
    return std::make_tuple(samples, sample_times);
}


#endif //HOPS_SAMPLERS_HPP
