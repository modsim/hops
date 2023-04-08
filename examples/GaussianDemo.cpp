#include <any>
#include <iostream>
#include <iomanip>

#include "hops/hops.hpp"
#include "hops/Parallel/MpiInitializerFinalizer.hpp"


std::tuple<Eigen::MatrixXd, Eigen::VectorXd> createSimplex(size_t dims) {
    Eigen::MatrixXd A(dims + 1, dims);
    Eigen::VectorXd upperBounds(dims);
    for (size_t i = 0; i < dims; ++i) {
        upperBounds(i) = 1.;
    }
    A << upperBounds.transpose(), -Eigen::MatrixXd::Identity(dims, dims);

    Eigen::VectorXd b(dims + 1);
    b << 5, Eigen::VectorXd::Zero(dims);

    return std::make_tuple(A, b);
}

int main() {
    std::srand(42);
    const long numberOfSamples = 2'00;
    const long thinning = 1;
    const long numberOfCheckPoints = 5;

    double covScale = 1e-2;
    const size_t dimStep = 2;
    Eigen::MatrixXd baseCov = covScale * Eigen::MatrixXd::Random(dimStep, dimStep);
    const Eigen::MatrixXd baseCovariance = baseCov.transpose() * baseCov + covScale * Eigen::MatrixXd::Identity(dimStep, dimStep);;

    const size_t maxDims = 3;
    std::vector<
            std::tuple<
                    Eigen::MatrixXd,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    hops::Gaussian,
                    hops::MarkovChainType,
                    long,
                    long,
                    double,
                    double,
                    bool,
                    std::string>
    > runConfigurations;

    for (size_t dims = dimStep; dims < maxDims; dims += dimStep) {
        Eigen::MatrixXd A(2 * dims, dims);
        A << Eigen::MatrixXd::Identity(dims, dims), -Eigen::MatrixXd::Identity(dims, dims);

        Eigen::VectorXd b(2 * dims);
        b << 5 * Eigen::VectorXd::Ones(dims), Eigen::VectorXd::Zero(dims);


        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dims);
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dims, dims);
        for (size_t i = 0; i < dims; i += dimStep) {
            covariance.block(i, i, dimStep, dimStep) = baseCovariance;
        }

        hops::Gaussian model(mean, covariance);
        mean = Eigen::VectorXd::Ones(dims);

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::AdaptiveMetropolis,
                                       numberOfSamples, thinning, 0, 0.23, true, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::BilliardAdaptiveMetropolis,
                                       numberOfSamples, thinning, 0, 0.23, true, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::BilliardMALA,
                                       numberOfSamples, thinning, 0, 0.5, false, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CSmMALA,
                                       numberOfSamples, thinning, 0, 0.5, false, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CSmMALA,
                                       numberOfSamples, thinning, 0.5, 0.5, false, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CoordinateHitAndRun,
                                       numberOfSamples, thinning, 0, 0.23, true, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::DikinWalk,
                                       numberOfSamples, thinning, 0, 0.23, false, "hypercube" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::HitAndRun,
                                       numberOfSamples, thinning, 0, 0.23, true, "hypercube" + std::to_string(A.cols()));
    }

    for (size_t dims = dimStep; dims < maxDims; dims += dimStep) {
        auto[A, b] = createSimplex(dims);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dims);
        mean(0) = 5;
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dims, dims);
        for (size_t i = 0; i < dims; i += dimStep) {
            covariance.block(i, i, dimStep, dimStep) = baseCovariance;
        }

        hops::Gaussian model(mean, covariance);
        mean = 1. / (dims + 10) * Eigen::VectorXd::Ones(dims);
        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::AdaptiveMetropolis,
                                       numberOfSamples, thinning, 0, 0.23, true, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::BilliardAdaptiveMetropolis,
                                       numberOfSamples, thinning, 0, 0.23, true, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::BilliardMALA,
                                       numberOfSamples, thinning, 0, 0.5, false, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CSmMALA,
                                       numberOfSamples, thinning, 0, 0.5, false, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CSmMALA,
                                       numberOfSamples, thinning, 0.5, 0.5, false, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::CoordinateHitAndRun,
                                       numberOfSamples, thinning, 0, 0.23, true, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::DikinWalk,
                                       numberOfSamples, thinning, 0, 0.23, false, "simplex" + std::to_string(A.cols()));

        runConfigurations.emplace_back(A, b, mean, model, hops::MarkovChainType::HitAndRun,
                                       numberOfSamples, thinning, 0, 0.23, true, "simplex" + std::to_string(A.cols()));
    }

    hops::MpiInitializerFinalizer::initializeAndQueueFinalizeAtExit();
    int numberOfChains;
    MPI_Comm_size(MPI_COMM_WORLD, &numberOfChains);
    int chainIndex;
    MPI_Comm_rank(MPI_COMM_WORLD, &chainIndex);

    for (size_t i = 0; i < runConfigurations.size(); i++) {
        if (i % numberOfChains != chainIndex) {
            continue;
        }
        const auto &run = runConfigurations[i];
        std::cout << "process " << chainIndex << "/" << numberOfChains << " is running " << i << ": " <<
                  hops::markovChainTypeToFullString(std::get<4>(run)) <<
                  " for " << std::get<9>(run) << " in " << std::get<0>(run).cols() << " dimensions" << std::endl;

        hops::Sampling::run(
                std::get<0>(run),
                std::get<1>(run),
                std::get<2>(run),
                std::get<3>(run),
                std::get<4>(run),
                std::get<5>(run),
                numberOfCheckPoints,
                std::get<6>(run),
                std::get<7>(run),
                std::get<8>(run),
                std::get<9>(run),
                std::get<10>(run));

    }
    std::cout << "finished: process " << chainIndex << "/" << numberOfChains << std::endl;
}
