#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>

int main() {
    auto Arounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/A_simplex_64D_rounded.csv");
    auto Nrounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../resources/simplex_64D/N_simplex_64D_rounded.csv");
    auto brounded = hops::CsvReader::readVector<Eigen::VectorXd>("../resources/simplex_64D/b_simplex_64D_rounded.csv");
    auto srounded = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/start_simplex_64D_rounded.csv");
    auto shift = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/p_shift_simplex_64D_rounded.csv");
    auto Aunrounded = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
            "../resources/simplex_64D/A_simplex_64D_unrounded.csv");
    auto bunrounded = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/b_simplex_64D_unrounded.csv");
    auto sunrounded = hops::CsvReader::readVector<Eigen::VectorXd>(
            "../resources/simplex_64D/start_simplex_64D_unrounded.csv");

    auto coordinateHitAndRun = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::CoordinateHitAndRun,
            Arounded,
            brounded,
            srounded,
            Nrounded,
            shift);

    auto hitAndRun = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::HitAndRun,
            Arounded,
            brounded,
            srounded,
            Nrounded,
            shift);

    auto dikinWalk = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::DikinWalk,
            Aunrounded,
            bunrounded,
            sunrounded);


    hops::RandomNumberGenerator randomNumberGenerator(42);

    coordinateHitAndRun->draw(randomNumberGenerator, 10000, 64);
    hitAndRun->draw(randomNumberGenerator, 10000, 64);
    dikinWalk->draw(randomNumberGenerator, 10000, 64);

    coordinateHitAndRun->writeHistory(
            hops::FileWriterFactory::createFileWriter("chrr", hops::FileWriterType::Csv).get());
    hitAndRun->writeHistory(hops::FileWriterFactory::createFileWriter("hrr", hops::FileWriterType::Csv).get());
    dikinWalk->writeHistory(hops::FileWriterFactory::createFileWriter("dikin", hops::FileWriterType::Csv).get());
}
