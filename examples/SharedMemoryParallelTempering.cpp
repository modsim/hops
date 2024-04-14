#include "hops/FileReader/CsvReader.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/MarkovChain/MarkovChainFactory.hpp"
#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/Model/Mixture.hpp"
#include "hops/MarkovChain/Proposal/Mixins/ParallelTemperingMixin.hpp"
#include "hops/MarkovChain/Proposal/ParallelTemperingImplementations/ParallelTemperingBoost.hpp"

int main() {
    Eigen::MatrixXd A(65, 64);
    A << -Eigen::MatrixXd::Identity(64, 64), Eigen::VectorXd::Ones(64).transpose();
    Eigen::VectorXd b = 100 * Eigen::VectorXd::Ones(65);
    Eigen::VectorXd mean1 = 5 * Eigen::VectorXd::Ones(64);
    Eigen::VectorXd mean2 = -5 * Eigen::VectorXd::Ones(64);
    Eigen::MatrixXd covariance = 1 * Eigen::MatrixXd::Identity(64, 64);

    hops::RandomNumberGenerator randomNumberGenerator(42);
    hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);

    auto model1 = std::make_shared<hops::Gaussian>(mean1, covariance);
    auto model2 = std::make_shared<hops::Gaussian>(mean2, covariance);
    hops::Coldness multimodalModel(hops::Mixture({model1, model2}));

    std::vector<std::unique_ptr<hops::MarkovChain>> chains;
    size_t n_chains = 20;
    size_t n_communicating_chains = 5;

    for(size_t i=0;i<n_chains;++i) {
        hops::VectorType mean = i%2==0 ? mean1 : mean2;
        hops::GaussianProposal prop(A, b, mean);
        hops::ModelMixin<decltype(prop), decltype(multimodalModel)> mprop(prop, multimodalModel);
        size_t chainIndex = i % n_communicating_chains;
        int replicate = int(double(i) / n_communicating_chains);
        std::cout << "chain " << i << " is part of replicate " << replicate << std::endl;
        std::string memory_name = "replicate_" + std::to_string(replicate) + "_";
        hops::ParallelTemperingBoost parallelTemperingImpl(n_communicating_chains, chainIndex, memory_name.c_str());
        hops::ParallelTemperingMixin<decltype(mprop), decltype(parallelTemperingImpl)> parallelTemperedProp(mprop, parallelTemperingImpl);
//        hops::MetropolisHastingsFilter<decltype(parallelTemperedProp)> mh(parallelTemperedProp);
//        auto mc = std::make_unique<hops::MarkovChainAdapter<decltype(mh)>>(mh);
//        chains.push_back(std::move(mc));
    }




//    auto markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
//            hops::MarkovChainType::CoordinateHitAndRun,
//            A,
//            b,
//            s,
//            multimodalModel,
//            synchronizedRandomNumberGenerator
//    );

//    long thinning = 10 * 64;
//    long numberOfSamples = 10000;
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
