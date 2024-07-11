#include "hops/hops.hpp"
#include <chrono>
#include <numeric>
#include <tuple>


int main() {
    const long numberOfSamples = 10'000;
    const long thinning = 5;
    const long numberOfCheckPoints = 5;
    const std::size_t dims = 3;

    double covScale = 1e-2;
    Eigen::MatrixXd baseCov = covScale * Eigen::MatrixXd::Random(dims, dims);
    // makes covariance positive definite
    const Eigen::MatrixXd covariance = baseCov.transpose() * baseCov + covScale * Eigen::MatrixXd::Identity(dims, dims);
    ;


    Eigen::MatrixXd A(2 * dims, dims);
    A << Eigen::MatrixXd::Identity(dims, dims), -Eigen::MatrixXd::Identity(dims, dims);

    Eigen::VectorXd b(2 * dims);
    b << 5 * Eigen::VectorXd::Ones(dims), Eigen::VectorXd::Zero(dims);


    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dims);

    hops::Gaussian model(mean, covariance);
    mean = Eigen::VectorXd::Ones(dims);

    // Rounding the polytope using the maximum volue ellipsoid is recommended to deal with anistropy.
    auto MVE = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 100000);
    auto startPointFinder = hops::LinearProgramFactory::createLinearProgram(A, b);
    hops::LinearProgramSolution startPointSolution = startPointFinder->computeChebyshevCenter();
    assert(startPointSolution.status == hops::LinearProgramStatus::OPTIMAL);
    Eigen::VectorXd startPoint = startPointSolution.optimalParameters;
    Eigen::MatrixXd roundingTrafo = MVE.getRoundingTransformation();
    Eigen::VectorXd startPointRounded = roundingTrafo.template triangularView<Eigen::Lower>().solve(startPoint);
    Eigen::MatrixXd Arounded = A * roundingTrafo;

    hops::MarkovChainType chainType = hops::MarkovChainType::CoordinateHitAndRun;

    auto gaussianSampler = hops::MarkovChainFactory::createMarkovChain(
            chainType,
            Arounded,
            b,
            startPointRounded,
            roundingTrafo,
            // we skipped shifting the polytopes chebyshev center to the origin in this example
            Eigen::VectorXd(Eigen::VectorXd::Zero(roundingTrafo.cols())),
            model);

    auto uniformSampler = hops::MarkovChainFactory::createMarkovChain(
            chainType,
            Arounded,
            b,
            startPointRounded,
            roundingTrafo,
            // we skipped shifting the polytopes chebyshev center to the origin in this example
            Eigen::VectorXd(Eigen::VectorXd::Zero(roundingTrafo.cols())));

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    std::vector<double> acceptanceRates;
    std::vector<Eigen::VectorXd> states;
    std::vector<long> timestamps;

    acceptanceRates.reserve(numberOfSamples);
    states.reserve(numberOfSamples);
    timestamps.reserve(numberOfSamples);


    auto uniform_results_writer = hops::FileWriterFactory::createFileWriter("uniform_sampling_results", hops::FileWriterType::CSV);
    auto gaussian_results_writer = hops::FileWriterFactory::createFileWriter("gaussian_sampling_results", hops::FileWriterType::CSV);


    // draws uniform samples
    for (long checkPoint = 0; checkPoint < numberOfCheckPoints; ++checkPoint) {
        for (long i = 0; i < numberOfSamples / numberOfCheckPoints; ++i) {
            ABORTABLE
            auto [acceptanceRate, state] = uniformSampler->draw(randomNumberGenerator, thinning);
            acceptanceRates.emplace_back(acceptanceRate);
            states.emplace_back(state);
            timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::high_resolution_clock::now().time_since_epoch())
                                            .count());
        }

        uniform_results_writer->write("states", states);
        uniform_results_writer->write("acceptance_rates", std::vector<double>{
                                                                  std::reduce(acceptanceRates.begin(), acceptanceRates.end(), 0.) / acceptanceRates.size()});
        uniform_results_writer->write("timestamps", timestamps);

        states.clear();
        acceptanceRates.clear();
        timestamps.clear();
    }

    // draws gaussian samples
    for (long checkPoint = 0; checkPoint < numberOfCheckPoints; ++checkPoint) {
        for (long i = 0; i < numberOfSamples / numberOfCheckPoints; ++i) {
            ABORTABLE
            auto [acceptanceRate, state] = gaussianSampler->draw(randomNumberGenerator, thinning);
            acceptanceRates.emplace_back(acceptanceRate);
            states.emplace_back(state);
            timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::high_resolution_clock::now().time_since_epoch())
                                            .count());
        }

        gaussian_results_writer->write("states", states);
        gaussian_results_writer->write("acceptance_rates", std::vector<double>{
                                                                   std::reduce(acceptanceRates.begin(), acceptanceRates.end(), 0.) / acceptanceRates.size()});
        gaussian_results_writer->write("timestamps", timestamps);

        states.clear();
        acceptanceRates.clear();
        timestamps.clear();
    }
}
