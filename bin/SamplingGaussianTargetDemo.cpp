#include <Eigen/SparseCore>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/Tuning/BinarySearchAcceptanceRateTuner.hpp>
#include <hops/Polytope/NormalizePolytope.hpp>
#include <iostream>
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>
#include <hops/MarkovChain/Tuning/AcceptanceRateTuner.hpp>

using RealType = double;

int main(int argc, char **argv) {
    if (argc != 11 && argc != 10) {
        std::cout << "usage: SamplingGaussianTarget A.csv b.csv mean.csv covariance.csv "
                  << "numberOfSamples thinningNumber CHRR|HRR|DikinWalk outputName [startingPoint.csv]"
                  << "\nArgument Description:\n"
                  << "\tA.csv\t\t\t\t nxm dimensional matrix of polytope Ax<b\n"
                  << "\tb.csv\t\t\t\t n dimensional vector of polytope Ax<b\n"
                  << "\tmean.csv\t\t\t m dimensional vector\n"
                  << "\tcovariance.csv\t\t mxm dimensional matrix\n"
                  << "\tnumberOfSamples\t\t number of samples to generate\n"
                  << "\tthinningNumber\t\t number of markov chain iterations per sample\n"
                  << "\talgorithm\t\t\t CHRR or HRR or DikinWalk\n"
                  << "\toutputName\t\t\t name for output\n"
                  << "\tfisherweight\t\t\t fisherweight\n"
                  << "\t[startingPoint]\t\t optional starting point, useful for resuming sampling" << std::endl;
        exit(0);
    }

    Eigen::SparseMatrix<RealType> A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            argv[1]).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> b = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            argv[2]).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> mean = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            argv[3]).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> covariance = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
            argv[4]).cast<RealType>();
    long numberOfSamples = std::strtol(argv[5], NULL, 10);
    long thinning = std::strtol(argv[6], NULL, 10);
    std::string chainName = argv[7];

    hops::MultivariateGaussianModel model(mean, covariance);

    std::shared_ptr<hops::MarkovChain> markovChain;

    if (chainName == "DikinWalk") {
        hops::MarkovChainType chainType = hops::MarkovChainType::DikinWalk;
        decltype(b) startingPoint;
        if (argc == 11) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[10]).cast<RealType>();
        } else {
            std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(A.cast<double>()),
                    b.cast<double>());
            startingPoint = linearProgram->computeChebyshevCenter().optimalParameters.cast<RealType>();
        }

        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model);
    } else if (chainName == "CSmMALA") {
        hops::MarkovChainType chainType = hops::MarkovChainType::CSmMALA;
        decltype(b) startingPoint;
        if (argc == 11) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[10]).cast<RealType>();
        } else {
            std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(A.cast<double>()),
                    b.cast<double>());
            startingPoint = linearProgram->computeChebyshevCenter().optimalParameters.cast<RealType>();
        }

        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model);

        double fisherWeight = std::stod(argv[9]);
        std::cout << "setting fisherweight to " << fisherWeight << std::endl;
        markovChain->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, fisherWeight);
    } else if (chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "CHRR" ? hops::MarkovChainType::CoordinateHitAndRun : hops::MarkovChainType::HitAndRun;
        Eigen::MatrixXd roundingTransformation = hops::MaximumVolumeEllipsoid<double>::construct(
                A,
                b,
                50000, 1e-9).getRoundingTransformation();

        decltype(b) startingPoint;
        if (argc == 11) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[10]).cast<RealType>();
        } else {
            std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>((A * roundingTransformation).cast<double>()),
                    b.cast<double>());
            startingPoint = linearProgram->computeChebyshevCenter().optimalParameters.cast<RealType>();
        }
        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>, decltype(b), decltype(model)>(
                chainType,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A * roundingTransformation),
                b,
                startingPoint,
                roundingTransformation,
                decltype(startingPoint)::Zero(roundingTransformation.rows()),
                model);
    } else {
        std::cerr << "No chain with chainname " << chainName << std::endl;
        std::exit(1);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    double lowerLimitStepSize = 1e-8;
    double upperLimitStepSize = 2;

    size_t iterationsToTestStepSize = 50;
    size_t posteriorUpdateIterations = 50;
    size_t pureSamplingIterations = 10;
    size_t stepSizeGridSize = 200;
    size_t iterationsForConvergence = 5;
    double smoothingLength = 0.001;
    bool recordData = true;

    if (chainName == "CSmMALA") {
        lowerLimitStepSize = 0.005;
        posteriorUpdateIterations = 10;
        stepSizeGridSize = 10;
        iterationsForConvergence = 5;
        smoothingLength = 0.005;
    }

    hops::AcceptanceRateTuner::param_type tuningParameters(0.234,
                                                           iterationsToTestStepSize,
                                                           posteriorUpdateIterations,
                                                           pureSamplingIterations,
                                                           iterationsForConvergence,
                                                           stepSizeGridSize,
                                                           lowerLimitStepSize,
                                                           upperLimitStepSize,
                                                           smoothingLength,
                                                           std::random_device()(),
                                                           recordData);

    std::vector<decltype(markovChain)> tuningChains = {markovChain};
    std::vector<decltype(randomNumberGenerator)> randomNumberGenerators = {randomNumberGenerator};
    hops::AcceptanceRateTuner::tune(tuningChains,
                                    randomNumberGenerators,
                                    tuningParameters);

    std::cout << "Current step size: " << markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE) << std::endl;

    auto fileWriter = hops::FileWriterFactory::createFileWriter(std::string(argv[8]) + "_" + markovChain->getName(),
                                                                hops::FileWriterType::CSV);
    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
