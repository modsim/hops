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


int main(int argc, char** argv) {
    if(argc!=4) {
        std::cout << "usage: ... chainIndex numChains replicate" << std::endl;
        exit(0);
    }
    int chainIndex = std::atoi(argv[1]);
    int n_communicating_chains = std::atoi(argv[2]);
    int replicate = std::atoi(argv[3]);

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

    std::vector<std::future<std::vector<hops::VectorType>>> samplers;
    long n_samples = 10000;
    std::vector<std::vector<hops::VectorType>> all_samples;

    hops::VectorType mean = chainIndex % 2 == 0 ? mean1 : mean2;
    hops::GaussianProposal prop(A, b, mean);
    hops::ModelMixin<decltype(prop), decltype(multimodalModel)> mprop(prop, multimodalModel);
    std::cout << "chain " << chainIndex << " is part of replicate " << replicate << std::endl;
    std::string memory_name = "replicate_" + std::to_string(replicate) + "_";
    hops::ParallelTemperingSEOBoostInterprocess parallelTemperingImpl(
            synchronizedRandomNumberGenerator,
            n_communicating_chains,
            chainIndex,
            memory_name.c_str(),
            mean.rows());
    hops::ParallelTemperingMixin<decltype(mprop), decltype(parallelTemperingImpl)> parallelTemperedProp(mprop,
                                                                                                        parallelTemperingImpl);
    hops::MetropolisHastingsFilter<decltype(parallelTemperedProp)> mh(parallelTemperedProp);
    auto mc = std::make_shared<hops::MarkovChainAdapter<decltype(mh)>>(mh);
    hops::RandomNumberGenerator rng(42 + chainIndex);

    std::vector<hops::VectorType> samples;

    for(long i=0; i<n_samples; ++i) {
        samples.emplace_back(mc->draw(rng).second);
    }

    hops::VectorType sample_mean = hops::VectorType::Zero(mean1.rows());
    for (size_t i = 0; i < samples.size(); ++i) {
        sample_mean += samples[i];
    }
    sample_mean /= n_samples;
    std::cout << chainIndex << " sample mean " << sample_mean.transpose() << std::endl;
}
