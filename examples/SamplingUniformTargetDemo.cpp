#include <Eigen/SparseCore>
#include <iostream>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <hops/PolytopePreprocessing/NormalizePolytope.hpp>

using RealType = double;

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: SamplingGaussianTargetDemo working_dir model_name chain_type" << std::endl
                  << "Example: ../../resources/ e_coli_core CHRR" << std::endl << std::endl
                  << "Arguments:" << std::endl
                  << "working_dir\t\t" << "directory containing the directory with the model." << std::endl
                  << "model_name\t\t" << "name of model" << std::endl
                  << "chain_type\t\t" << "CHRR | HRR | DikinWalk" << std::endl;
        exit(0);
    }
    std::string modelDirectory(argv[1]);
    std::string modelName(argv[2]);
    std::string chainName(argv[3]);
    std::cout << "model directory " << modelDirectory << std::endl;
    std::cout << "model name " << modelName << std::endl;
    std::cout << "chain name " << chainName << std::endl;

    std::string Afile = modelDirectory + "/" + modelName + "/A_" + modelName + "_unrounded.csv";
    std::string Tfile = modelDirectory + "/" + modelName + "/T_" + modelName + "_rounded.csv";
    std::string bfile = modelDirectory + "/" + modelName + "/b_" + modelName + "_unrounded.csv";
    std::string startFile = modelDirectory + "/" + modelName + "/start_" + modelName + "_unrounded.csv";

    std::cout << "A file " << Afile << std::endl;
    std::cout << "b file " << bfile << std::endl;

    Eigen::SparseMatrix<RealType> A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            Afile).cast<RealType>();
    Eigen::SparseMatrix<RealType> roundingTransformation = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            Tfile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> b = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            bfile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> startingPoint = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            startFile).cast<RealType>();

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    std::unique_ptr<hops::MarkovChain> markovChain;
    if (chainName == "DikinWalk") {
        hops::MarkovChainType chainType = hops::MarkovChainType::DikinWalk;
        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A),
                                                                  b,
                                                                  startingPoint);
        float upperLimitAcceptanceRate = 0.3;
        float lowerLimitAcceptanceRate = 0.20;
        double lowerLimitStepSize = 1e-15;
        double upperLimitStepSize = 1;
        size_t iterationsToTestStepSize = 10 * A.cols();
        size_t maxIterations = 10000 * A.cols();
        for (int i = 0; i < 5; ++i) {
            markovChain->draw(randomNumberGenerator, 1000, A.cols() * 10);
            markovChain->clearHistory();
            markovChain->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 0.075);

            bool isTuned = hops::AcceptanceRateTuner::tune(markovChain.get(),
                                                           randomNumberGenerator,
                                                           {lowerLimitAcceptanceRate,
                                                            upperLimitAcceptanceRate,
                                                            lowerLimitStepSize,
                                                            upperLimitStepSize,
                                                            iterationsToTestStepSize,
                                                            maxIterations});
            std::cout << "isTuned: " << isTuned << std::endl;
            std::cout << "current step size: " << markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE)
                      << std::endl;
            markovChain->clearHistory();
        }
    } else if (chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "HRR" ? hops::MarkovChainType::HitAndRun : hops::MarkovChainType::CoordinateHitAndRun;

        // Assumes rounding transformation (the result of a cholesky decomposition) is stored as lower diagonal L of LLT and not UUT
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> roundedStartingPoint = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(
                roundingTransformation)
                .template triangularView<Eigen::Lower>().solve(startingPoint);

        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>, decltype(b)>(
                chainType,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A * roundingTransformation),
                b,
                roundedStartingPoint,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(roundingTransformation),
                decltype(roundedStartingPoint)::Zero(roundingTransformation.rows()));
    } else {
        std::cerr << "No chain with chainname " << chainName << std::endl;
        std::exit(1);
    }

    auto fileWriter = hops::FileWriterFactory::createFileWriter(
            modelName + "_" + markovChain->getName() + "_uniform_demo",
            hops::FileWriterType::Csv);
    long thinning = A.cols() * 100;
    long numberOfSamples = 100;
    for (int i = 0; i < 100; ++i) {
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
        markovChain->writeHistory(fileWriter.get());
        markovChain->clearHistory();
    }
}