#include <algorithm>
#include <thread>
#include <future>

#include "hops/FileReader/CsvReader.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/MarkovChain/MarkovChainFactory.hpp"
#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/Model/Mixture.hpp"
#include "hops/MarkovChain/Proposal/Mixins/ParallelTemperingMixin.hpp"
#include "hops/MarkovChain/Proposal/ParallelTemperingImplementations/ParallelTemperingSEOBoostInterprocess.hpp"


std::vector<hops::VectorType> sample_chain(std::shared_ptr<hops::MarkovChain> chain,
                                           long n_samples,
                                           hops::RandomNumberGenerator &rng) {
    std::vector<hops::VectorType> samples;
    samples.reserve(n_samples);
    for (long i = 0; i < n_samples; ++i) {
        auto [_, sample] = chain->draw(rng);
        samples.emplace_back(sample);
    }
    return samples;
}

int main() {
    long dims = 2;
    Eigen::MatrixXd A(2*dims, dims);
    A << -Eigen::MatrixXd::Identity(dims, dims), Eigen::MatrixXd::Identity(dims,dims);
    Eigen::VectorXd b = 100 * Eigen::VectorXd::Ones(2*dims);
    Eigen::VectorXd mean1 = 5 * Eigen::VectorXd::Ones(dims);
    Eigen::VectorXd mean2 = -5 * Eigen::VectorXd::Ones(dims);
    Eigen::MatrixXd covariance = 1 * Eigen::MatrixXd::Identity(dims, dims);

    hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);

    auto model1 = std::make_shared<hops::Gaussian>(mean1, covariance);
    auto model2 = std::make_shared<hops::Gaussian>(mean2, covariance);
    hops::Coldness multimodalModel(hops::Mixture({model1, model2}));

//    std::vector<std::thread> samplers;
    std::vector<std::future<std::vector<hops::VectorType>>> samplers;
    size_t n_chains = 1;
    size_t n_communicating_chains = 1;
    long n_samples = 10000;
    std::vector<std::vector<hops::VectorType>> all_samples;

    for (size_t i = 0; i < n_chains; ++i) {
        hops::VectorType mean = i % 2 == 0 ? mean1 : mean2;
        hops::GaussianProposal prop(A, b, mean);
        hops::ModelMixin<decltype(prop), decltype(multimodalModel)> mprop(prop, multimodalModel);
        size_t chainIndex = i % n_communicating_chains;
        int replicate = int(double(i) / n_communicating_chains);
        std::cout << "chain " << i << " is part of replicate " << replicate << std::endl;
        std::string memory_name = "replicate_" + std::to_string(replicate) + "_";
        hops::ParallelTemperingSEOBoostInterprocess parallelTemperingImpl(synchronizedRandomNumberGenerator, n_communicating_chains, chainIndex, memory_name.c_str());
        hops::ParallelTemperingMixin<decltype(mprop), decltype(parallelTemperingImpl)> parallelTemperedProp(mprop,
                                                                                                            parallelTemperingImpl);
        hops::MetropolisHastingsFilter<decltype(parallelTemperedProp)> mh(parallelTemperedProp);
//        hops::MetropolisHastingsFilter<decltype(mprop)> mh(mprop);
        auto mc = std::make_shared<hops::MarkovChainAdapter<decltype(mh)>>(mh);
        hops::RandomNumberGenerator rng(42 + i);
        std::vector<hops::VectorType> samples;
//        all_samples.push_back(samples);
//        std::thread sampler(sample_chain, mc, std::ref(samples), n_samples, std::ref(rng));
        auto sampler = std::async(sample_chain, mc, n_samples, std::ref(rng));
        samplers.push_back(std::move(sampler));
    }

    for (auto &t: samplers) {
        hops::VectorType mean = hops::VectorType::Zero(mean1.rows());
        auto res = t.get();
        for (size_t i = 0; i < res.size(); ++i) {
            mean += res[i];
        }
        mean /= n_samples;
        std::cout << "mean " << mean.transpose() << std::endl;
    }

//    for (auto &s: all_samples) {
//        for (size_t i = 0; i > s.size(); ++i) {
//            mean += s[i];
//        }
//        mean /= n_samples;
//        std::cout << "mean " << mean.transpose() << std::endl;
//    }

//    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
//            std::chrono::high_resolution_clock::now().time_since_epoch()
//    ).count();
//
//    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
//
//    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
//            std::chrono::high_resolution_clock::now().time_since_epoch()
//    ).count();
//    std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
//              << " seconds, that's "
//              << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
//              << " s per sample"
//              << std::endl;
//
//    auto fileWriter = hops::FileWriterFactory::createFileWriter("parallelTemperingDemo", hops::FileWriterType::CSV);
//    markovChain->writeHistory(fileWriter.get());
}
