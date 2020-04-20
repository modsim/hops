#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/ParallelTempering/ParallelTempering.hpp>
#include <hops/MarkovChain/ParallelTempering/ColdnessAttribute.hpp>
#include <hops/Model/MultimodalModel.hpp>

/**
 * @brief Run with mpiexec
 * @return
 */
int main() {
    Eigen::MatrixXd A(16, 8);
    A << Eigen::MatrixXd::Identity(8, 8), -Eigen::MatrixXd::Identity(8, 8);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(16);
    Eigen::VectorXd s = Eigen::VectorXd::Zero(8);
    Eigen::VectorXd mean1 = Eigen::VectorXd::Ones(8);
    Eigen::VectorXd mean2 = -Eigen::VectorXd::Ones(8);
    Eigen::MatrixXd covariance = 0.075 * Eigen::MatrixXd::Identity(8, 8);
    hops::RandomNumberGenerator randomNumberGenerator(42);

    hops::MultivariateGaussianModel model1(mean1, covariance);
    hops::MultivariateGaussianModel model2(mean2, covariance);
    hops::ColdnessAttribute multimodalModel(hops::MultimodalModel(std::make_tuple(model1, model2)));

    double exchangeAttemptProbability = 0.5;
    auto markovChain = hops::MarkovChainAdapter(
            hops::ParallelTempering(
                    hops::StateRecorder(
                            hops::MetropolisHastingsFilter(
                                    hops::ModelMixin(
                                            hops::CoordinateHitAndRunProposal<
                                                    Eigen::MatrixXd,
                                                    Eigen::VectorXd,
                                                    hops::UniformStepDistribution<double>>(
                                                    A,
                                                    b,
                                                    s), multimodalModel
                                    )
                            )
                    ),
                    exchangeAttemptProbability
            )
    );

    long thinning = 100;
    long numberOfSamples = 10000;
    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();

    markovChain.draw(randomNumberGenerator, numberOfSamples, thinning);

    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
              << " s per sample"
              << std::endl;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto fileWriter = hops::FileWriterFactory::createFileWriter("parallelTemperingDemo" + std::to_string(rank), hops::FileWriterType::Csv);
    markovChain.writeHistory(fileWriter.get());
}
