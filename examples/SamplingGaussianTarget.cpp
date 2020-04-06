#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/Model/Model.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>

int main() {
    auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto Arounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_rounded.csv");
    auto Nrounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/N_simplex_64D_rounded.csv");
    auto Trounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/T_simplex_64D_rounded.csv");
    auto brounded = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_rounded.csv");
    auto srounded = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/start_simplex_64D_rounded.csv");
    auto shiftrounded = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv");

    auto fileWriter1 = hops::FileWriterFactory::createFileWriter("chain1", hops::FileWriterType::Csv);
    auto fileWriter2 = hops::FileWriterFactory::createFileWriter("chain2", hops::FileWriterType::Csv);
    auto fileWriter3 = hops::FileWriterFactory::createFileWriter("chain3", hops::FileWriterType::Csv);
    auto fileWriter4 = hops::FileWriterFactory::createFileWriter("chain4", hops::FileWriterType::Csv);

    hops::RandomNumberGenerator randomNumberGenerator(42);

    const Eigen::VectorXd &mean = s;
    Eigen::MatrixXd covariance = Trounded.transpose() * Trounded;

    auto markovChain1 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::Model(
                                    hops::DikinProposal(
                                            A, b, s
                                    ),
                                    hops::MultivariateGaussianModel(mean, covariance)
                            )
                    )
            )
    );
    markovChain1.setStepSize(3. / 40);

    auto markovChain2 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::Model(
                                    hops::StateTransformation(
                                            hops::HitAndRunProposal<
                                                    Eigen::MatrixXd,
                                                    Eigen::VectorXd,
                                                    hops::GaussianStepDistribution<double>>(
                                                    Arounded,
                                                    brounded,
                                                    srounded),
                                            hops::Transformation(Nrounded, shiftrounded)),
                                    hops::MultivariateGaussianModel(mean, covariance))
                    )
            )
    );

    auto markovChain3 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::Model(
                                    hops::StateTransformation(
                                            hops::CoordinateHitAndRunProposal<
                                                    Eigen::MatrixXd,
                                                    Eigen::VectorXd,
                                                    hops::GaussianStepDistribution<double>>(
                                                    Arounded,
                                                    brounded,
                                                    srounded),
                                            hops::Transformation(Nrounded, shiftrounded)),
                                    hops::MultivariateGaussianModel(mean, covariance))
                    )
            )
    );

    auto markovChain4 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::CSmMALAProposal(
                                    hops::MultivariateGaussianModel(mean, covariance), A, b, s)
                    )
            )
    );
    markovChain4.setStepSize(3. / 40);

    long thinning = 256;
    long numberOfSamples = 10000;
    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    markovChain1.draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain2.draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain3.draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain4.draw(randomNumberGenerator, numberOfSamples, thinning);
    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
              << " s per sample"
              << std::endl;
    markovChain1.writeHistory(fileWriter1.get());
    markovChain2.writeHistory(fileWriter2.get());
    markovChain3.writeHistory(fileWriter3.get());
    markovChain4.writeHistory(fileWriter4.get());
    markovChain1.clearHistory();
    markovChain2.clearHistory();
    markovChain3.clearHistory();
    markovChain4.clearHistory();
}

