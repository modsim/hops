#include <Eigen/SparseCore>
#include <hops/hops.hpp>
#include <iostream>
#include <random>
#include <hops/Model/LinearModel.hpp>
#include <iomanip>


int main(int argc, char **argv) {

    auto simplex = hops::SimplexFactory<Eigen::MatrixXd, Eigen::VectorXd>::createSimplex(8);
    Eigen::MatrixXd A = std::get<0>(simplex);
    Eigen::VectorXd b = std::get<1>(simplex);
    std::uniform_real_distribution<double> uniformRealDistribution(-10, 10);

    double stddev = 0.4;
    std::normal_distribution<double> normalDistribution(0, stddev);

    hops::RandomNumberGenerator randomNumberGenerator(3);

    Eigen::MatrixXd model = Eigen::MatrixXd::Zero(6, 8);
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 7; j++) {
            model(i, j) = uniformRealDistribution(randomNumberGenerator);
        }
    }

    Eigen::VectorXd trueParams = 1. / 8 * Eigen::VectorXd::Ones(8);

    Eigen::VectorXd measuredData = model * trueParams;
    for (long i = 0; i < measuredData.rows(); ++i) {
        measuredData(i) += normalDistribution(randomNumberGenerator);
    }

    hops::LinearModel<Eigen::MatrixXd, Eigen::VectorXd> target(measuredData, stddev * stddev *
                                                                             Eigen::MatrixXd::Identity(
                                                                                     measuredData.rows(),
                                                                                     measuredData.rows()), model);

    std::cout << model << std::endl;
    std::cout << measuredData.transpose() << std::endl;


    std::vector<std::string> argumentNames = {
            "outputIdentifier",
            "startingPoint",
            "randomSeed",
            "numberOfSamples",
            "thinningFactor",
            "numberOfCheckpoints",
            "markovChainType",
            "fisherWeight"
    };

    std::vector<std::string> argumentDescriptions = {
            "identifier string that uniquely identifies this run",
            "int that references to which corner to start from",
            "seed for rng",
            "number of samples per checkpoint",
            "applies number of parameters times thinning factor as thinning",
            "number of checkpoints (at least 1, because the samples are only saved at check points)",
            "BallWalk | CSmMALA | CHRR | HRR | DikinWalk ",
            "fisherWeight"
    };

    if (argc != 1 + static_cast<int>(argumentNames.size())) {
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

    std::map<std::string, std::string> programArguments;
    for (size_t i = 0; i < argumentNames.size(); ++i) {
        programArguments.insert({argumentNames[i], argv[i + 1]});
    };


    std::unique_ptr<hops::MarkovChain> markovChain;


    std::string randomSeed = programArguments.find("randomSeed")->second;
    std::string markovChainName = programArguments.find("markovChainType")->second;
    std::string uniqueIdentifier = programArguments.find("outputIdentifier")->second;
    std::string fisherWeight = programArguments.find("fisherWeight")->second;
    long startingPointSource = std::stol(programArguments.find("startingPoint")->second);
    Eigen::VectorXd startingPoint = Eigen::VectorXd::Zero(trueParams.rows());
    for (long i = 0; i < startingPoint.rows(); ++i) {
        startingPoint(i) = 1. / (startingPoint.rows() * 100);
    }
    startingPoint(startingPointSource) = 1 - static_cast<double>(startingPoint.rows()-1)/startingPoint.rows()/50;

    std::cout << "starting point is interior: " << ((b - A * startingPoint).array() > 0).all() << std::endl;


    if (markovChainName == "CHRR" || markovChainName == "HRR") {
        hops::MarkovChainType chainType = markovChainName == "CHRR" ? hops::MarkovChainType::CoordinateHitAndRun :
                                          hops::MarkovChainType::HitAndRun;

        auto roundingTransformation = hops::MaximumVolumeEllipsoid<double>::construct(
                A, b, 10000).getRoundingTransformation();

        if (!roundingTransformation.isLowerTriangular()) {
            throw std::runtime_error("Error while rounding starting point, check code.");
        }
        std::cout << "original start " << startingPoint.transpose() << std::endl;
        startingPoint = roundingTransformation.triangularView<Eigen::Lower>().solve(startingPoint);
        std::cout << "start is " << startingPoint.transpose() << std::endl;


        if (((b - A * roundingTransformation * startingPoint).array() <= 0).any()) {
            std::cout << "rounded starting point is not valid" << std::endl;
            throw std::runtime_error("error deriving rounded start point");
        }

        markovChain = hops::MarkovChainFactory::createMarkovChain(
                chainType,
                Eigen::MatrixXd(A * roundingTransformation),
                b,
                startingPoint,
                Eigen::MatrixXd(roundingTransformation),
                Eigen::VectorXd(Eigen::VectorXd::Zero(startingPoint.rows())),
                target
        );
    } else if (markovChainName == "CSmMALA") {
        std::cout << "start is " << startingPoint.transpose() << std::endl;
        markovChain = hops::MarkovChainFactory::createMarkovChain(
                hops::MarkovChainType::CSmMALA,
                A,
                b,
                startingPoint,
                target
        );
        markovChain->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, std::stod(fisherWeight));
    } else if (markovChainName == "DikinWalk") {
        std::cout << "start is " << startingPoint.transpose() << std::endl;
        markovChain = hops::MarkovChainFactory::createMarkovChain(
                hops::MarkovChainType::DikinWalk,
                A,
                b,
                startingPoint,
                target
        );
    }

    std::string fileWriterOutput =  "linearModel_" + uniqueIdentifier + "_" + markovChainName +
                                   (markovChainName == "CSmMALA" ? +"_" + fisherWeight : "");
    auto fileWriter = hops::FileWriterFactory::createFileWriter(fileWriterOutput, hops::FileWriterType::CSV);

    try {
        double upperLimitAcceptanceRate = 0.23;
        double lowerLimitAcceptanceRate = 0.20;

        double lowerLimitStepSize = 1e-10;
        double upperLimitStepSize = 2;

        // Counts how often markov chain could be tuned in a row
        bool isTuned = false;
        int countIsTuned = 0;
        // Limits how long a single tuning run should last
        size_t iterationsToTestStepSize = 100;
        size_t maxIterations = iterationsToTestStepSize * 1000;

        // Tuning loop
        for (int i = 0; (i < 10) && (countIsTuned < 4); ++i) {
            markovChain->draw(randomNumberGenerator, 1, 1000);
            markovChain->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);
            markovChain->clearHistory();

            isTuned = hops::AcceptanceRateTuner::tune(markovChain.get(),
                                                      randomNumberGenerator,
                                                      {lowerLimitAcceptanceRate,
                                                       upperLimitAcceptanceRate,
                                                       lowerLimitStepSize,
                                                       upperLimitStepSize,
                                                       iterationsToTestStepSize,
                                                       maxIterations});
            markovChain->draw(randomNumberGenerator, 1, 1000);
            std::cout << "step size at end of loop " << std::endl <<
                      markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE)
                      << std::endl << "acceptance rate " << std::endl << markovChain->getAcceptanceRate() << std::endl;

            markovChain->clearHistory();
            countIsTuned = isTuned * (countIsTuned + 1);
        }
        dynamic_cast <const hops::CsvWriter *>(fileWriter.get())->write("tuningStatus",
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
    std::cout << "start is " << startingPoint.transpose() << std::endl;


    long thinning = std::stol(programArguments.find("thinningFactor")->second);
    long numberOfSamples = std::stol(programArguments.find("numberOfSamples")->second);
    long numberOfCheckpoints = std::stol(programArguments.find("numberOfCheckpoints")->second);
    for (int i = 0; i < numberOfCheckpoints; ++i) {
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
        markovChain->writeHistory(fileWriter.get());
        markovChain->clearHistory();
    }
}
