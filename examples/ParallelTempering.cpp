#include "hops/FileReader/CsvReader.hpp"
#include "hops/FileWriter/FileWriterFactory.hpp"
#include "hops/MarkovChain/MarkovChainFactory.hpp"
#include "hops/MarkovChain/ModelMixin.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/Model/Mixture.hpp"

/**
 * @brief Run with mpiexec
 * @details not supported on windows
 * @return
 */
int main() {
    Eigen::MatrixXd A(65, 64);
    A << -Eigen::MatrixXd::Identity(64, 64), Eigen::VectorXd::Ones(64).transpose();
    Eigen::VectorXd b = Eigen::VectorXd::Ones(65);
    Eigen::VectorXd s = 1. / 100 * Eigen::VectorXd::Ones(64);
    Eigen::VectorXd mean1 = 1. / 1000 * Eigen::VectorXd::Ones(64);
    Eigen::VectorXd mean2 = 1. / 200 * Eigen::VectorXd::Ones(64);
    Eigen::MatrixXd covariance = 0.075 * Eigen::MatrixXd::Identity(64, 64);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    hops::RandomNumberGenerator randomNumberGenerator(42, world_rank);
    hops::RandomNumberGenerator synchronizedRandomNumberGenerator(42);

    hops::MultivariateGaussianModel model1(mean1, covariance);
    hops::MultivariateGaussianModel model2(mean2, covariance);
    hops::ColdnessAttribute multimodalModel(hops::MultimodalModel(std::make_tuple(model1, model2)));

    auto markovChain = hops::MarkovChainFactory::createMarkovChainWithParallelTempering(
            hops::MarkovChainType::CoordinateHitAndRun,
            A,
            b,
            s,
            multimodalModel,
            synchronizedRandomNumberGenerator
    );

    long thinning = 10 * 64;
    long numberOfSamples = 10000;
    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);

    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
              << " s per sample"
              << std::endl;

    auto fileWriter = hops::FileWriterFactory::createFileWriter("parallelTemperingDemo", hops::FileWriterType::CSV);
    markovChain->writeHistory(fileWriter.get());
}
