#include <nups/FileReader/CsvReader.hpp>
#include <nups/FileWriter/FileWriterFactory.hpp>
#include <nups/MarkovChain/MarkovChainFactory.hpp>
#include <nups/PolytopeSpace/PolytopeSpace.hpp>

int main() {
    auto A = nups::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto b = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto s = nups::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto fileWriter = nups::FileWriterFactory::createFileWriter("simplex_64D_unrounded", nups::FileWriterType::Csv);

    nups::RandomNumberGenerator randomNumberGenerator(42);

    nups::PolytopeSpace polytopeSpace(A, b, s);

    auto markovChain = nups::MarkovChainFactory::createMarkovChain(polytopeSpace,
                                                                   nups::MarkovChainType::CoordinateHitAndRun);
    markovChain->draw(randomNumberGenerator, 10000, 10000);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
