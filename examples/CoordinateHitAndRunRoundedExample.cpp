#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/PolytopeSpace/PolytopeSpace.hpp>
#include <hops/Transformation/Transformation.hpp>

int main() {
    auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_rounded.csv");
    auto N = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/N_simplex_64D_rounded.csv");
    auto T = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/T_simplex_64D_rounded.csv");
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_rounded.csv");
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_rounded.csv");
    auto shift = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv");

    auto fileWriter = hops::FileWriterFactory::createFileWriter("simplex_64D_rounded", hops::FileWriterType::Csv);

    hops::RandomNumberGenerator randomNumberGenerator(42);

    hops::PolytopeSpace roundedPolytopeSpace(A, b, s);
    hops::Transformation roundingTransformation(N, shift);

    auto markovChain = hops::MarkovChainFactory::createMarkovChain(roundedPolytopeSpace,
                                                                   hops::MarkovChainType::CoordinateHitAndRunRoundedStateSpace);

    markovChain->draw(randomNumberGenerator, 1000, 100000);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
