#include <nups/FileReader/CsvReader.hpp>
#include <nups/FileWriter/FileWriterFactory.hpp>
#include <nups/MarkovChain/MarkovChainFactory.hpp>
#include <nups/PolytopeSpace/PolytopeSpace.hpp>
#include <nups/MarkovChain/Recorder/StateRecorder.hpp>
#include <nups/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <nups/MarkovChain/Proposal/DikinProposal.hpp>
#include <nups/Model/Model.hpp>
#include <nups/Model/MultivariateGaussianModel.hpp>
#include <nups/MarkovChain/Proposal/CSmMALAProposal.hpp>

int main() {
//    Eigen::MatrixXd A(4, 2);
//    A << -1, 0, 0, -1, 1, 0, 0, 1;
//    Eigen::VectorXd b = Eigen::VectorXd::Zero(4);
//    b(b.rows() - 1) = 1;
//    b(b.rows() - 2) = 1;
//    Eigen::VectorXd s(2);
//    s << 0.2, 0.2;
    auto A = nups::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto b = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto s = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto fileWriter1 = nups::FileWriterFactory::createFileWriter("chain1", nups::FileWriterType::Csv);
    auto fileWriter2 = nups::FileWriterFactory::createFileWriter("chain2", nups::FileWriterType::Csv);
    auto fileWriter3 = nups::FileWriterFactory::createFileWriter("chain3", nups::FileWriterType::Csv);

    nups::RandomNumberGenerator randomNumberGenerator(42);

    nups::PolytopeSpace polytopeSpace(A, b, s);

    Eigen::VectorXd mean = s;
    Eigen::MatrixXd covariance = 0.00005*Eigen::MatrixXd::Identity(A.cols(), A.cols());

    auto markovChain1 = nups::MarkovChainAdapter(
            nups::StateRecorder(
                    nups::MetropolisHastingsFilter(
                            nups::Model(
                                    nups::DikinProposal(
                                            A, b, s
                                    ),
                                    nups::MultivariateGaussianModel(mean, covariance)
                            )
                    )
            )
    );

    auto markovChain2 = nups::MarkovChainAdapter(
            nups::StateRecorder(
                    nups::MetropolisHastingsFilter(
                            nups::Model(
                                    nups::DikinProposal(
                                            A, b, s
                                    ),
                                    nups::MultivariateGaussianModel(mean, covariance)
                            )
                    )
            )
    );

    auto markovChain3 = nups::MarkovChainAdapter(
            nups::StateRecorder(
                    nups::MetropolisHastingsFilter(
                            nups::CSmMALAProposal(
                                    nups::MultivariateGaussianModel(mean, covariance), A, b, s)
                    )
            )
    );

    long thinning = 64;
    long numberOfSamples = 10000;
    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    markovChain1.draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain2.draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain3.draw(randomNumberGenerator, numberOfSamples, thinning);
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
    markovChain1.clearHistory();
    markovChain2.clearHistory();
    markovChain3.clearHistory();
}

