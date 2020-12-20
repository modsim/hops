#include <Eigen/SparseCore>
#include <iostream>
#include <iomanip>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <hops/Polytope/NormalizePolytope.hpp>
#include <hops/Model/RosenbrockModel.hpp>

int main(int argc, char **argv) {
    std::vector<std::string> argumentNames = {
            "dimensions",
            "scale_factor",
            "number_of_samples",
            "thinning_factor",
            "output_name",
            "chain_type",
    };

    std::vector<std::string> argumentDescriptions = {
            "number of dimensions, has to be divisible by 2",
            "scale_factor",
            "how many samples to generate into output file",
            "applies number of parameters times thinning factor as thinning",
            "name of directory that results will be written to",
            "CHRR | HRR | DikinWalk | CSmMALA | CSmMALANoGradient"
    };

    if (static_cast<size_t>(argc) != argumentNames.size() + 1) {
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
    };

    long numberOfDimensions = std::stol(programArguments.find("dimensions")->second);
    double scaleFactor = std::stod(programArguments.find("scale_factor")->second);
    long numberOfSamples = std::stol(programArguments.find("number_of_samples")->second);
    long thinningFactor = std::stol(programArguments.find("thinning_factor")->second);
    std::string outputName(programArguments.find("output_name")->second);
    std::string chainName(programArguments.find("chain_type")->second);


    Eigen::MatrixXd A(numberOfDimensions*2, numberOfDimensions);
    A << Eigen::MatrixXd::Identity(numberOfDimensions, numberOfDimensions), -Eigen::MatrixXd::Identity(numberOfDimensions, numberOfDimensions);
    Eigen::MatrixXd roundingTransformation = Eigen::MatrixXd::Identity(numberOfDimensions, numberOfDimensions);
    Eigen::VectorXd b = 10 * Eigen::VectorXd::Ones(numberOfDimensions * 2);
    Eigen::VectorXd start = -7.5 * Eigen::VectorXd::Ones(numberOfDimensions);

    auto model = hops::RosenbrockModel<Eigen::MatrixXd, Eigen::VectorXd>(scaleFactor, Eigen::VectorXd::Zero(numberOfDimensions / 2));

    std::unique_ptr<hops::MarkovChain> markovChain;

    if (chainName == "DikinWalk" || chainName == "CSmMALA" || chainName == "CSmMALANoGradient") {
        hops::MarkovChainType chainType =
                chainName == "DikinWalk" ? hops::MarkovChainType::DikinWalk :
                chainName == "CSmMALA" ? hops::MarkovChainType::CSmMALA
                                       : hops::MarkovChainType::CSmMALANoGradient;

        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  start,
                                                                  model,
                                                                  false);
    }
    else if (chainName == "BallWalk" || chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "BallWalk" ? hops::MarkovChainType::BallWalk :
                chainName == "HRR" ? hops::MarkovChainType::HitAndRun : hops::MarkovChainType::CoordinateHitAndRun;
        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::MatrixXd, decltype(b), decltype(model)>(
                chainType,
                Eigen::MatrixXd(A),
                b,
                start,
                model,
                false);
    }
    else {
        std::cerr << "No chain with chainName " << chainName << std::endl;
        std::exit(1);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    auto fileWriter = hops::FileWriterFactory::createFileWriter(
            outputName + "_" + markovChain->getName(),
            hops::FileWriterType::Csv);


    try {
        double upperLimitAcceptanceRate = 0.24;
        double lowerLimitAcceptanceRate = 0.22;

        double lowerLimitStepSize = 1e-16;
        double upperLimitStepSize = 2;

        // Counts how often markov chain could be tuned in a row
        bool isTuned = false;
        // Limits how long a single tuning run should last
        size_t iterationsToTestStepSize = 20 * A.cols();
        size_t maxIterations = 20000 * A.cols();

        // Tuning loop
        for (int i = 0; i < 2; ++i) {
            markovChain->draw(randomNumberGenerator, 1, 2000);
            markovChain->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);
            markovChain->resetAcceptanceRate();

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


    long thinning = thinningFactor;
    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}

