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

using RealType = double;

int main(int argc, char **argv) {
    if (argc != 10 && argc != 9) {
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

    std::unique_ptr<hops::MarkovChain> markovChain;
    if (chainName == "DikinWalk") {
        hops::MarkovChainType chainType = hops::MarkovChainType::DikinWalk;
        decltype(b) startingPoint;
        if (argc == 10) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[9]).cast<RealType>();
        } else {
            std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(A.cast<double>()),
                    b.cast<double>();
            startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();
        }

        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model);
    } else if (chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "CHRR" ? hops::MarkovChainType::CoordinateHitAndRun : hops::MarkovChainType::HitAndRun;
        Eigen::MatrixXd roundingTransformation = hops::MaximumVolumeEllipsoid<double>::construct(
                A,
                b,
                50000, 1e-9).getRoundingTransformation();

        decltype(b) startingPoint;
        if (argc == 10) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[9]).cast<RealType>();
        } else {
            std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>((A * roundingTransformation).cast<double>()),
                    b.cast<double>());
            startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();
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

    float upperLimitAcceptanceRate = 0.3;
    float lowerLimitAcceptanceRate = 0.20;
    double lowerLimitStepSize = 1e-15;
    double upperLimitStepSize = 1;
    size_t iterationsToTestStepSize = 100 * A.cols();
    size_t maxIterations = 10000 * A.cols();

    bool isTuned = false;
    // Tuning loop
    for (int i = 0; i < 10; ++i) {
        markovChain->draw(randomNumberGenerator, 1, numberOfSamples);
        markovChain->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);

        isTuned = hops::AcceptanceRateTuner::tune(markovChain.get(),
                                                  randomNumberGenerator,
                                                  {lowerLimitAcceptanceRate,
                                                   upperLimitAcceptanceRate,
                                                   lowerLimitStepSize,
                                                   upperLimitStepSize,
                                                   iterationsToTestStepSize,
                                                   maxIterations});
        markovChain->clearHistory();
    }
    std::cout << "Markov chain tuned successfully : " << std::boolalpha << isTuned
              << " (false is not a problem for CHRR|HRR)" << std::endl;
    std::cout << "Current step size: " << markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE) << std::endl;

    auto fileWriter = hops::FileWriterFactory::createFileWriter(std::string(argv[8]) + "_" + markovChain->getName(),
                                                                hops::FileWriterType::CSV);
    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
