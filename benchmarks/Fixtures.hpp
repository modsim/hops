#ifndef HOPS_FIXTURES_HPP
#define HOPS_FIXTURES_HPP

#include <celero/Celero.h>
#include "hops/FileReader/CsvReader.hpp"
#include "hops/MarkovChain/MarkovChainFactory.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/Transformation/Transformation.hpp"
#include "PolytopeSpace.hpp"

template<typename ModelFiles, typename Matrix, typename Vector>
class PolytopeSpaceFixture {
public:
    PolytopeSpaceFixture() {
        ModelFiles modelFiles;
        polytopeSpace.A = hops::CsvReader::readMatrix<Matrix>(modelFiles.A);
        polytopeSpace.b = hops::CsvReader::readVector<Vector>(modelFiles.b);
        polytopeSpace.startingPoint = hops::CsvReader::readVector<Vector>(modelFiles.startingPoint);

        polytopeSpace.roundedA = hops::CsvReader::readMatrix<Matrix>(modelFiles.roundedA);
        polytopeSpace.roundedb = hops::CsvReader::readVector<Vector>(modelFiles.roundedb);
        polytopeSpace.roundedStartingPoint = hops::CsvReader::readVector<Vector>(modelFiles.roundedStartingPoint);
        polytopeSpace.roundedN = hops::CsvReader::readMatrix<Matrix>(modelFiles.roundedN);
        polytopeSpace.roundedShift = hops::CsvReader::readVector<Vector>(modelFiles.roundedp_shift);
    }

    hops::PolytopeSpace<Matrix, Vector> polytopeSpace;
};

template<typename PolytopeSpaceFixture, hops::MarkovChainType markovChainType>
class MarkovChainFixture : public celero::TestFixture {
public:
    MarkovChainFixture() {
        Eigen::VectorXd mean = polytopeSpaceFixture.polytopeSpace.startingPoint;
        Eigen::MatrixXd covariance = 0.01 * mean.asDiagonal();
        hops::MultivariateGaussianModel model(mean, covariance);
        markovChain = hops::MarkovChainFactory::createMarkovChain(
                markovChainType,
                polytopeSpaceFixture.polytopeSpace.A,
                polytopeSpaceFixture.polytopeSpace.b,
                polytopeSpaceFixture.polytopeSpace.startingPoint,
                model,
                false
        );
    }

    PolytopeSpaceFixture polytopeSpaceFixture;
    std::unique_ptr<hops::MarkovChain> markovChain;
    hops::RandomNumberGenerator randomNumberGenerator{std::random_device{}()};
};

template<char ... characters>
struct ModelFiles {
    ModelFiles() {
        name = ((std::string(1, characters) + ... ));
        A = std::string("../resources/") + name + std::string("/A_") + name +
            std::string("_unrounded.csv");
        b = std::string("../resources/") + name + std::string("/b_") + name +
            std::string("_unrounded.csv");
        startingPoint = std::string("../resources/") + name + std::string("/start_") + name +
                        std::string("_unrounded.csv");
        roundedA = std::string("../resources/") + name + std::string("/A_") + name +
                   std::string("_rounded.csv");
        roundedb = std::string("../resources/") + name + std::string("/b_") + name +
                   std::string("_rounded.csv");
        roundedStartingPoint = std::string("../resources/") + name + std::string("/start_") + name +
                               std::string("_rounded.csv");
        roundedN = std::string("../resources/") + name + std::string("/N_") + name +
                   std::string("_rounded.csv");
        roundedp_shift = std::string("../resources/") + name + std::string("/p_shift_") + name +
                         std::string("_rounded.csv");
    }

    std::string name;
    std::string A;
    std::string b;
    std::string startingPoint;
    std::string roundedA;
    std::string roundedb;
    std::string roundedStartingPoint;
    std::string roundedN;
    std::string roundedp_shift;
};

using simplex_64D_files = ModelFiles<'s', 'i', 'm', 'p', 'l', 'e', 'x', '_', '6', '4', 'D'>;
using simplex_256D_files = ModelFiles<'s', 'i', 'm', 'p', 'l', 'e', 'x', '_', '2', '5', '6', 'D'>;
using simplex_512D_files = ModelFiles<'s', 'i', 'm', 'p', 'l', 'e', 'x', '_', '5', '1', '2', 'D'>;
using simplex_1024D_files = ModelFiles<'s', 'i', 'm', 'p', 'l', 'e', 'x', '_', '1', '0', '2', '4', 'D'>;
using simplex_2048D_files = ModelFiles<'s', 'i', 'm', 'p', 'l', 'e', 'x', '_', '2', '0', '4', '8', 'D'>;
using e_coli_core_files = ModelFiles<'e', '_', 'c', 'o', 'l', 'i', '_', 'c', 'o', 'r', 'e'>;
using iAT_PLT_636_files = ModelFiles<'i', 'A', 'T', '_', 'P', 'L', 'T', '_', '6', '3', '6'>;
using iJO1366_files = ModelFiles<'i', 'J', 'O', '1', '3', '6', '6'>;
using Recon1_files = ModelFiles<'R', 'E', 'C', 'O', 'N', '1'>;
using Recon2_files = ModelFiles<'R', 'e', 'c', 'o', 'n', '2', '.', 'v', '0', '4'>;

#endif //HOPS_FIXTURES_HPP
