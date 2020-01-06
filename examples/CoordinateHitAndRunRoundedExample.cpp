#include <nups/FileReader/CsvReader.hpp>
#include <nups/FileWriter/FileWriterFactory.hpp>
#include <nups/MarkovChain/MarkovChainFactory.hpp>
#include <nups/PolytopeSpace/PolytopeSpace.hpp>
#include <nups/Transformation/Transformation.hpp>

int main() {
    auto A = nups::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_rounded.csv");
    auto N = nups::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/N_simplex_64D_rounded.csv");
    auto T = nups::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/T_simplex_64D_rounded.csv");
    auto b = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_rounded.csv");
    auto s = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_rounded.csv");
    auto shift = nups::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv");

    auto fileWriter = nups::FileWriterFactory::createFileWriter("simplex_64D_rounded", nups::FileWriterType::Csv);

    nups::RandomNumberGenerator randomNumberGenerator(42);

    nups::PolytopeSpace roundedPolytopeSpace(A, b, s);
    nups::Transformation roundingTransformation(N, shift);

    auto markovChain = nups::MarkovChainFactory::createMarkovChain(roundedPolytopeSpace,
                                                                   nups::MarkovChainType::CoordinateHitAndRunRoundedStateSpace);

    markovChain->draw(randomNumberGenerator, 1000, 100000);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
