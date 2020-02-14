#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/PolytopeSpace/PolytopeSpace.hpp>

int main() {
    auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto fileWriter = hops::FileWriterFactory::createFileWriter("simplex_64D_unrounded", hops::FileWriterType::Csv);

    hops::RandomNumberGenerator randomNumberGenerator(42);

    hops::PolytopeSpace polytopeSpace(A, b, s);

    auto markovChain = hops::MarkovChainFactory::createMarkovChain(polytopeSpace,
                                                                   hops::MarkovChainType::CoordinateHitAndRun);
    markovChain->draw(randomNumberGenerator, 10000, 10000);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
