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
#include <hops/PolytopePreprocessing/NormalizePolytope.hpp>
#include <hops/MarkovChain/Recorder/MetropolisHastingsInfoRecorder.hpp>

using RealType = double;

int main(int argc, char **argv) {
    std::vector<std::string> argumentNames = {
            "workingDir",
            "model_name",
            "mean_prefix",
            "covariance_prefix",
            "number_of_samples",
            "thinning_factor",
            "output_name",
            "chain_type",
    };

    std::vector<std::string> argumentDescriptions = {
            "base dir which contains the model directory",
            "name of the model. A directory named after the model_name is expected in the workingDir. See ../../resources/e_coli_core for an example",
            "prefix for mean. workingDir/mean_prefix_model_name.csv will be read in",
            "prefix for covariance. workingDir/covariance_prefix_model_name.csv will be read in",
            "how many samples to generate into output file",
            "applies number of parameters times thinning factor as thinning",
            "name of directory that results will be written to",
            "CHRR | HRR | DikinWalk | CSmMALA | CSmMALANoGradient"
    };

    if (argc != 9) {
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
            {argumentNames[7], argv[8]},
    };

    std::string workingDir(programArguments.find("workingDir")->second);
    std::string modelName(programArguments.find("model_name")->second);
    std::string meanPrefix(programArguments.find("mean_prefix")->second);
    std::string covariancePrefix(programArguments.find("covariance_prefix")->second);
    long numberOfSamples = std::stol(programArguments.find("number_of_samples")->second);
    long thinningFactor = std::stol(programArguments.find("thinning_factor")->second);
    std::string outputName(programArguments.find("output_name")->second);
    std::string chainName(programArguments.find("chain_type")->second);

    std::cout << "Looking for model directory in: " << workingDir << std::endl;
    std::cout << "Model name: " << modelName << std::endl;
    std::cout << "Using: " << chainName << std::endl;
    std::cout << "Writing results to: ./" << outputName << std::endl;

    std::string aFile = workingDir + "/" + modelName + "/A_" + modelName + "_unrounded.csv";
    std::string tFile = workingDir + "/" + modelName + "/T_" + modelName + "_rounded.csv";
    std::string bfile = workingDir + "/" + modelName + "/b_" + modelName + "_unrounded.csv";
    std::string meanFile = workingDir + "/" + modelName + "/" + meanPrefix + "_" + modelName + ".csv";
    std::string startFile = workingDir + "/" + modelName + "/start_" + modelName + "_unrounded.csv";
    std::string covarianceFile =
            workingDir + "/" + modelName + "/" + covariancePrefix + "_" + modelName + ".csv";
    std::string parameterNamesFile = workingDir + "/" + modelName + "/parameterNames_" + modelName + ".csv";

    std::cout << "Reading A matrix from " << aFile << std::endl;
    std::cout << "Reading b vector from " << bfile << std::endl;
    std::cout << "Reading starting vector from " << startFile << std::endl;
    std::cout << "Rounding matrix from " << tFile << std::endl;
    std::cout << "Reading mean vector from " << meanFile << std::endl;
    std::cout << "Reading covariance matrix from " << covarianceFile << std::endl;
    std::cout << "Reading parameter names from " << parameterNamesFile << std::endl;

    Eigen::SparseMatrix<RealType> A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            aFile).cast<RealType>();
    Eigen::SparseMatrix<RealType> roundingTransformation = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            tFile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> b = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            bfile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> start = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            startFile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> mean = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            meanFile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> covariance = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
            covarianceFile).cast<RealType>();
    auto parameterNames = hops::CsvReader::readVector<std::vector<std::string>>(parameterNamesFile);

    auto model = hops::MetropolisHastingsInfoRecorder(hops::MultivariateGaussianModel(mean, covariance));

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
    } else if (chainName == "BallWalk" || chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "BallWalk" ? hops::MarkovChainType::BallWalk :
                chainName == "HRR" ? hops::MarkovChainType::HitAndRun : hops::MarkovChainType::CoordinateHitAndRun;

        // Assumes rounding transformation (the result of a cholesky decomposition) is stored as lower diagonal L of LLT and not UUT
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
    } else {
        std::cerr << "No chain with chainName " << chainName << std::endl;
        std::exit(1);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    auto fileWriter = hops::FileWriterFactory::createFileWriter(
            outputName + "_" + modelName + "_" + markovChain->getName(),
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

    dynamic_cast<const hops::CsvWriter *>(fileWriter.get())->write("parameterNames", parameterNames);
    for (const auto &argumentName: argumentNames) {
        std::string argumentNameAndValue = argumentName + ": " + programArguments.find(argumentName)->second;
        dynamic_cast<const hops::CsvWriter *>(fileWriter.get())->write("arguments", {argumentNameAndValue});
    }


    long thinning = A.cols() * thinningFactor;
    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}

