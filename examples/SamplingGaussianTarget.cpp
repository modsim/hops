#include <Eigen/SparseCore>
#include <chrono>
#include <iostream>
#include <filesystem>

#include "hops/hops.hpp"

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
            argv[4], true).cast<RealType>();
    long numberOfSamples = std::strtol(argv[5], NULL, 10);
    long thinning = std::strtol(argv[6], NULL, 10);
    std::string chainName = argv[7];

    hops::Gaussian model(mean, covariance);

    std::shared_ptr<hops::MarkovChain> markovChain;
    hops::MarkovChainType chainType = hops::stringToMarkovChainType(chainName);
    if (chainType == hops::MarkovChainType::BilliardMALA ||
        chainType == hops::MarkovChainType::CSmMALA ||
        chainType == hops::MarkovChainType::DikinWalk) {
        decltype(b) startingPoint;
        if (argc == 10) {
            startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
                    argv[9]).cast<RealType>();
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
    } else if (chainType == hops::MarkovChainType::CoordinateHitAndRun ||
               chainType == hops::MarkovChainType::HitAndRun) {
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

    double lowerLimitStepSize = 1./2;
    double upperLimitStepSize = 2;
    size_t iterationsToTestStepSize = 200;
    size_t posteriorUpdateIterations = 100;
    size_t pureSamplingIterations = 10;
    size_t stepSizeGridSize = std::log10(upperLimitStepSize / lowerLimitStepSize) * 10;
    size_t iterationsForConvergence = 5;
    double smoothingLength = 1;
    bool recordData = true;

    std::vector<std::shared_ptr<hops::MarkovChain>> tuningChains = {markovChain};
    std::vector<hops::RandomNumberGenerator *> randomNumberGenerators = {&randomNumberGenerator};
    double targetAcceptanceRate = 0.23;

    auto tuningTarget = hops::AcceptanceRateTarget(
            tuningChains,
            iterationsToTestStepSize,
            targetAcceptanceRate,
            2);


    hops::MatrixType data;
    double deltaAcceptanceRate = 1;
    hops::VectorType stepSize(1);
    double measuredAcceptanceRate = -1;
    long tuningIteration = 0;
    double tuningTolerance = 0.05;

    auto fileWriter = hops::FileWriterFactory::createFileWriter(std::string(argv[8]) + "_" + chainName,
                                                                hops::FileWriterType::CSV);


    while (deltaAcceptanceRate > tuningTolerance && tuningIteration < 5) {
        stepSize(0) = std::any_cast<double>(tuningChains[0]->getParameter(hops::ProposalParameter::STEP_SIZE));
        hops::ThompsonSamplingTuner::param_type tuningParameters(
                posteriorUpdateIterations,
                pureSamplingIterations,
                iterationsForConvergence,
                stepSizeGridSize,
                lowerLimitStepSize,
                upperLimitStepSize,
                smoothingLength,
                std::random_device()(),
                recordData);

        hops::ThompsonSamplingTuner::tune(
                stepSize,
                deltaAcceptanceRate,
                randomNumberGenerators,
                tuningParameters,
                tuningTarget,
                data);

        measuredAcceptanceRate = tuningChains[0]->draw(randomNumberGenerator, 200).first;
        deltaAcceptanceRate = std::abs(targetAcceptanceRate - measuredAcceptanceRate);

        std::stringstream stream;
        stream << "tuning iter: " << tuningIteration << " " << hops::markovChainTypeToShortString(chainType) << " s: "
               << stepSize(0) << " alpha: "
               << measuredAcceptanceRate
               << " (delta: " << deltaAcceptanceRate << ")" << " u: " << upperLimitStepSize << " l: "
               << lowerLimitStepSize << std::endl;

        fileWriter->write("tuning_debug_info", std::vector<std::string>{stream.str()});


        // Does not tune step size too high for our hit&run walks.
        if (chainType == hops::MarkovChainType::CoordinateHitAndRun || chainType == hops::MarkovChainType::HitAndRun) {
            if (upperLimitStepSize >= 10) {
                break;
            }
        }

        if (deltaAcceptanceRate > tuningTolerance && measuredAcceptanceRate < targetAcceptanceRate) {
            upperLimitStepSize /= 2;
            lowerLimitStepSize /= 2;
        }
        if (deltaAcceptanceRate > tuningTolerance && measuredAcceptanceRate > targetAcceptanceRate) {
            upperLimitStepSize *= 2;
            lowerLimitStepSize *= 2;
        }
        stepSizeGridSize += std::log10(upperLimitStepSize / lowerLimitStepSize);
        tuningIteration++;
    }

    std::vector<double> acceptanceRates;
    std::vector<Eigen::VectorXd> states;
    std::vector<long> timestamps;
    acceptanceRates.reserve(numberOfSamples);
    states.reserve(numberOfSamples);
    timestamps.reserve(numberOfSamples);

    for (int i = 0; i < numberOfSamples; ++i) {
        auto[acceptanceRate, state] = markovChain->draw(randomNumberGenerator, thinning);
        acceptanceRates.emplace_back(acceptanceRate);
        states.emplace_back(state);
        timestamps.emplace_back(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count());
    }


    fileWriter->write("states", states);
    fileWriter->write("acceptance_rates", acceptanceRates);
    fileWriter->write("timestamps", timestamps);
}
