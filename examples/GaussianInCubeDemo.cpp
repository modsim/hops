#include <Eigen/SparseCore>
#include <chrono>
#include <iostream>
#include <filesystem>

#include <hops/hops.hpp>


template<typename ModelType>
void runExperiment(const Eigen::MatrixXd &A,
                   const Eigen::VectorXd &b,
                   const Eigen::VectorXd &startPoint,
                   const ModelType &model,
                   hops::MarkovChainType chainType,
                   int numberOfSamples) {

    std::unique_ptr<hops::MarkovChain> markovChain = hops::MarkovChainFactory::createMarkovChain(
            chainType,
            A,
            b,
            startPoint,
            model);


    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    std::vector<double> acceptanceRates;
    std::vector<double> negLogLikelihoods;
    std::vector<Eigen::VectorXd> states;
    std::vector<long> timestamps;

    acceptanceRates.reserve(numberOfSamples);
    negLogLikelihoods.reserve(numberOfSamples);
    states.reserve(numberOfSamples);
    timestamps.reserve(numberOfSamples);

    std::unique_ptr<hops::FileWriter> writer;

    std::cout << "start sampling " << hops::MarkovChainTypeToFullString(chainType) << std::endl;

    for (int i = 0; i < numberOfSamples; ++i) {
        auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator, 1);
        acceptanceRates.emplace_back(acceptanceRate);
        negLogLikelihoods.emplace_back(markovChain->getStateNegativeLogLikelihood());
        states.emplace_back(state);
        timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count());
    }
    writer = hops::FileWriterFactory::createFileWriter(hops::MarkovChainTypeToFullString(chainType),
                                                       hops::FileWriterType::CSV);
    writer->write("states", states);
    writer->write("acceptance_rates", acceptanceRates);
    writer->write("neg_log_likelihoods", negLogLikelihoods);
    writer->write("timestamps", timestamps);
    states.clear();
    acceptanceRates.clear();
    negLogLikelihoods.clear();
    timestamps.clear();

    std::cout << "finish sampling " << hops::MarkovChainTypeToFullString(chainType) << std::endl;
}

int main() {
    int numberOfSamples = 20'000;

    Eigen::MatrixXd A(4, 2);
    A << 1, 0, 0, 1, -1, 0, 0, -1;
    Eigen::VectorXd b(4);
    b << 10, 10, 0, 0;
//    b << 10, 10, 10, 10;

    Eigen::VectorXd mean = Eigen::VectorXd::Zero(2);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Identity(2, 2);
    mean = 1e-10*Eigen::VectorXd::Ones(2);

    hops::Gaussian model(mean, covariance);

//    runExperiment(A, b, mean, model, hops::MarkovChainType::CoordinateHitAndRun,
//                  numberOfSamples);

    runExperiment(A, b, mean, model, hops::MarkovChainType::CSmMALA,
                  numberOfSamples);

//    runExperiment(A, b, mean, model, hops::MarkovChainType::DikinWalk,
//                  numberOfSamples);
//
//    runExperiment(A, b, mean, model, hops::MarkovChainType::Gaussian,
//                  numberOfSamples);
//
//    runExperiment(A, b, mean, model, hops::MarkovChainType::HitAndRun,
//                  numberOfSamples);
//
    runExperiment(A, b, mean, model, hops::MarkovChainType::BillardMALA,
                  numberOfSamples);
}
