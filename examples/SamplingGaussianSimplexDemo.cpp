#include <Eigen/SparseCore>
#include <iostream>
#include <iomanip>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <hops/Polytope/NormalizePolytope.hpp>
#include <hops/Polytope/SimplexFactory.hpp>
#include <hops/MarkovChain/Recorder/MetropolisHastingsInfoRecorder.hpp>
#include <hops/Polytope/MaximumVolumeEllipsoid.hpp>

using RealType = double;

int main(int argc, char **argv) {
    std::vector<std::string> argumentNames = {
            "number_of_dimensions",
            "mean",
            "covariance_scale",
            "number_of_samples",
            "thinning_factor",
            "chain_type",
            "s",
    };

    std::vector<std::string> argumentDescriptions = {
            "number of dimensions",
            "location of mean, either corner or chebyshev",
            "scale of covariance entries",
            "how many samples to generate into output file",
            "applies number of parameters times thinning factor as thinning",
            "CHRR | CSmMALA | CSmMALANoGradient | DikinWalk | HRR",
            "mixing parameter for CSmMALA"
    };

    if (argc != 8) {
        std::cout << "Usage: ";
        size_t maxArgumentNamesWidth = 0;
        for (const auto &argumentName: argumentNames) {
            std::cout << argumentName << " ";
            maxArgumentNamesWidth = std::max(maxArgumentNamesWidth, argumentName.size());
        }

        std::cout << std::endl << std::endl << "Arguments:" << std::endl;
        for (size_t i = 0; i < argumentNames.size(); ++i) {
            std::cout << std::setw(static_cast<int>(maxArgumentNamesWidth)) << std::left
                      << argumentNames[i] << "\t" << argumentDescriptions[i] << std::endl;
        }

        exit(1);
    }


    std::map<std::string, std::string> programArguments = {
            {argumentNames[0], argv[1]},
            {argumentNames[1], argv[2]},
            {argumentNames[2], argv[3]},
            {argumentNames[3], argv[4]},
            {argumentNames[4], argv[5]},
            {argumentNames[5], argv[6]},
            {argumentNames[6], argv[7]},
    };

    long numberOfDimensions = std::stol(programArguments.find("number_of_dimensions")->second);
    long numberOfSamples = std::stol(programArguments.find("number_of_samples")->second);
    long thinningFactor = std::stol(programArguments.find("thinning_factor")->second);
    std::string meanType(programArguments.find("mean")->second);
    double covarianceScale = std::stod(programArguments.find("covariance_scale")->second);
    std::string chainName(programArguments.find("chain_type")->second);
    double fisherWeight = std::stod(programArguments.find("s")->second);

    auto[A, b] = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(numberOfDimensions);

    Eigen::VectorXd start = hops::LinearProgramGurobiImpl(A, b).calculateChebyshevCenter().optimalParameters;

    Eigen::Matrix<RealType, Eigen::Dynamic, 1> mean;
    if (meanType == "corner") {
        mean = Eigen::VectorXd::Zero(numberOfDimensions);
        mean(0) = 1;
    } else if (meanType == "chebyshev") {
        mean = start;
    } else {
        throw std::runtime_error("undefined meantype '" + meanType + "', see help message.");
    }

    std::uniform_real_distribution<double> uniformRealDistribution;
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> covariance = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>::Zero(
            mean.rows(), mean.rows());
    for (long i = 0; i < mean.rows(); ++i) {
        covariance(i, i) = covarianceScale;
    }
    for (long i = 0; i < mean.rows(); ++i) {
        for (long j = 0; j < mean.rows(); ++j) {
            if (i != j) {
                covariance(i, j) = covarianceScale / 10;
            }
        }
    }
    covariance = covariance.transpose() * covariance;

    std::cout << "cov " << std::endl << covariance << std::endl;

    auto model = hops::MetropolisHastingsInfoRecorder(hops::MultivariateGaussianModel(mean, covariance));
    hops::MarkovChainType chainType;
    if (chainName == "BallWalk") {
        chainType = hops::MarkovChainType::BallWalk;
    } else if (chainName == "CHRR") {
        chainType = hops::MarkovChainType::CoordinateHitAndRun;
    } else if (chainName == "CSmMALA") {
        chainType = hops::MarkovChainType::CSmMALA;
    } else if (chainName == "CSmMALAOld") {
        chainType = hops::MarkovChainType::CSmMALAOld;
    } else if (chainName == "CSmMALANoGradient") {
        chainType = hops::MarkovChainType::CSmMALANoGradient;
    } else if (chainName == "DikinWalk") {
        chainType = hops::MarkovChainType::DikinWalk;
    } else if (chainName == "DikinWalkOld") {
        chainType = hops::MarkovChainType::DikinWalkOld;
    } else if (chainName == "HRR") {
        chainType = hops::MarkovChainType::HitAndRun;
    } else {
        throw std::runtime_error("");
    }

    std::unique_ptr<hops::MarkovChain> markovChain;
    if (chainType == hops::MarkovChainType::BallWalk ||
        chainType == hops::MarkovChainType::CSmMALA ||
        chainType == hops::MarkovChainType::CSmMALANoGradient ||
        chainType == hops::MarkovChainType::CSmMALAOld ||
        chainType == hops::MarkovChainType::DikinWalk ||
        chainType == hops::MarkovChainType::DikinWalkOld) {

        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  start,
                                                                  model,
                                                                  false);
        if (chainType == hops::MarkovChainType::CSmMALA || chainType == hops::MarkovChainType::CSmMALANoGradient) {
            markovChain->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, fisherWeight);
        }
    } else if (chainType == hops::MarkovChainType::CoordinateHitAndRun ||
               chainType == hops::MarkovChainType::HitAndRun) {
        // Assumes rounding transformation (the result of a cholesky decomposition)
        // is stored as lower diagonal L of LLT and not UUT
        Eigen::MatrixXd roundingTransformation = hops::MaximumVolumeEllipsoid<double>::construct(A, b,
                                                                                                 1e8).getRoundingTransformation();

        Eigen::Matrix<RealType, Eigen::Dynamic, 1> roundedStart = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(
                roundingTransformation)
                .template triangularView<Eigen::Lower>().solve(start);

        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>, decltype(b), decltype(model)>(
                chainType,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A * roundingTransformation),
                b,
                roundedStart,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(roundingTransformation),
                decltype(roundedStart)::Zero(roundingTransformation.rows()),
                model,
                false);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    auto fileWriter = hops::FileWriterFactory::createFileWriter(
            "simplex_" + std::to_string(numberOfDimensions) + "D" + "_mean_" + meanType + "_covariance_" +
            std::to_string(covarianceScale) + "_" + chainName + "_" + programArguments.find("s")->second,
            hops::FileWriterType::Csv);


    try {
        double upperLimitAcceptanceRate = 0.25;
        double lowerLimitAcceptanceRate = 0.20;

        double lowerLimitStepSize = 1e-5;
        double upperLimitStepSize = 1;

        // Counts how often markov chain could be tuned in a row
        bool isTuned = false;
        // Limits how long a single tuning run should last
        size_t iterationsToTestStepSize = 400;
        size_t maxIterations = 400 * iterationsToTestStepSize;

        // Tuning loop
        for (int i = 0; i < 4; ++i) {
            markovChain->draw(randomNumberGenerator, 1, 2000);
            markovChain->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);
            markovChain->resetAcceptanceRate();
            markovChain->clearHistory();
            isTuned = hops::AcceptanceRateTuner::tune(markovChain.get(),
                                                      randomNumberGenerator,
                                                      {lowerLimitAcceptanceRate,
                                                       upperLimitAcceptanceRate,
                                                       lowerLimitStepSize,
                                                       upperLimitStepSize,
                                                       iterationsToTestStepSize,
                                                       maxIterations});
            markovChain->draw(randomNumberGenerator, 1, 1);

            std::cout << "step size at end of loop " << std::endl <<
                      markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE) << std::endl <<
                      "acceptance rate " << std::endl <<
                      markovChain->getAcceptanceRate() << std::endl;
            markovChain->draw(randomNumberGenerator, 1, 2000);

            markovChain->resetAcceptanceRate();
            markovChain->clearHistory();
        }
        dynamic_cast<const hops::CsvWriter *>(fileWriter.get())->write("tuningStatus",
                                                                       {"1 for success, 0 for failure"});
        Eigen::VectorXd tuningResult(1);
        tuningResult(0) = isTuned;
        fileWriter->write("tuningStatus", tuningResult);
    }
    catch (std::runtime_error &e) {
        std::cout << "Skipping tuning (Reason: " << e.what() << ")" << std::endl;
    }
    Eigen::VectorXd stepSize(1);
    stepSize(0) = markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    fileWriter->write("stepSize", stepSize);

    for (const auto &argumentName: argumentNames) {
        std::string argumentNameAndValue = argumentName + ": " + programArguments.find(argumentName)->second;
        dynamic_cast<const hops::CsvWriter *>(fileWriter.get())->write("arguments", {argumentNameAndValue});
    }

    long thinning = A.cols() * thinningFactor;
    for (long i = 0; i < numberOfSamples; i += 1000) {
        markovChain->draw(randomNumberGenerator, 1000, thinning);
        markovChain->writeHistory(fileWriter.get());
        markovChain->clearHistory();
    }
}

