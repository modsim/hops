#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/StateTransformation.hpp>
#include <hops/MarkovChain/Proposal/CoordinateHitAndRunProposal.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/Transformation/Transformation.hpp>
#include <hops/MarkovChain/Draw/NoOpDrawAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>

int main() {
    auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_rounded.csv");
    auto N = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/N_simplex_64D_rounded.csv");
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_rounded.csv");
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_rounded.csv");
    auto shift = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv");
    auto A2 = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto b2 = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto s2 = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto markovChain1 = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::StateRecorder(
                            hops::StateTransformation(
                                    hops::CoordinateHitAndRunProposal(
                                            A,
                                            b,
                                            s),
                                    hops::Transformation(N, shift))
                    )
            )
    );

    auto markovChain2 = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::StateRecorder(
                            hops::StateTransformation(
                                    hops::HitAndRunProposal(
                                            A,
                                            b,
                                            s),
                                    hops::Transformation(N, shift))
                    )
            )
    );

    auto markovChain3 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::DikinProposal(
                                    A2,
                                    b2,
                                    s2)
                    )
            )
    );


    hops::RandomNumberGenerator randomNumberGenerator(42);

    markovChain1.draw(randomNumberGenerator, 10000, 64);
    markovChain2.draw(randomNumberGenerator, 10000, 64);
    markovChain3.draw(randomNumberGenerator, 10000, 64);

    markovChain1.writeHistory(hops::FileWriterFactory::createFileWriter("chrr", hops::FileWriterType::Csv).get());
    markovChain2.writeHistory(hops::FileWriterFactory::createFileWriter("hrr", hops::FileWriterType::Csv).get());
    markovChain3.writeHistory(hops::FileWriterFactory::createFileWriter("dikin", hops::FileWriterType::Csv).get());
}
