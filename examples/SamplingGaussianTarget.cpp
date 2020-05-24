#include <Eigen/SparseCore>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <hops/PolytopePreprocessing/NormalizePolytope.hpp>
#include <iostream>

using RealType = double;

int main(int argc, char **argv) {
    std::string modelDirectory(argv[1]);
    std::string modelName(argv[2]);
    std::string chainName(argv[3]);
    std::cout << "model directory " << modelDirectory << std::endl;
    std::cout << "model name " << modelName << std::endl;
    std::cout << "chain name " << chainName << std::endl;

    // TODO fix paths for for windows
    std::string Afile = modelDirectory + "/" + modelName + "/A_" + modelName + "_unrounded.csv";
    std::string Tfile = modelDirectory + "/" + modelName + "/T_" + modelName + "_rounded.csv";
    std::string bfile = modelDirectory + "/" + modelName + "/b_" + modelName + "_unrounded.csv";
    std::string meanFile = modelDirectory + "/" + modelName + "/start_" + modelName + "_unrounded.csv";

    std::cout << "A file " << Afile << std::endl;
    std::cout << "b file " << bfile << std::endl;

    Eigen::SparseMatrix<RealType> A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            Afile).cast<RealType>();
    Eigen::SparseMatrix<RealType> roundingTransformation = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(
            Tfile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> b = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            bfile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> mean = hops::CsvReader::readVector<Eigen::Matrix<double, Eigen::Dynamic, 1>>(
            meanFile).cast<RealType>();
    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> denseA = A;
    hops::normalizePolytope(denseA, b);
    A = denseA.sparseView();
    A.makeCompressed();

    Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> covariance =
            1e-6 * Eigen::Matrix<RealType, Eigen::Dynamic, 1>::Ones(mean.rows()).asDiagonal();
    hops::MultivariateGaussianModel model(mean, covariance);

    std::unique_ptr<hops::MarkovChain> markovChain;
    if (chainName == "DikinWalk") {
        std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(A.cast<double>()),
                b.cast<double>());
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();

        markovChain = hops::MarkovChainFactory::createMarkovChain(hops::MarkovChainType::DikinWalk,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model,
                                                                  false);
    } else if (chainName == "CSmMALA") {
        std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>(A.cast<double>()),
                b.cast<double>());
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();
        markovChain = hops::MarkovChainFactory::createMarkovChain(hops::MarkovChainType::CSmMALA,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model,
                                                                  false);
    } else if (chainName == "CHRR") {
        std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>((A * roundingTransformation).cast<double>()),
                b.cast<double>());
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();
        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>, decltype(b), decltype(model)>(
                hops::MarkovChainType::CoordinateHitAndRun,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A * roundingTransformation),
                b,
                startingPoint,
                roundingTransformation,
                decltype(startingPoint)::Zero(roundingTransformation.rows()),
                model,
                false);
    } else if (chainName == "HRR") {
        std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>((A * roundingTransformation).cast<double>()),
                b.cast<double>());
        Eigen::Matrix<RealType, Eigen::Dynamic, 1> startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters.cast<RealType>();
        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>, decltype(b), decltype(model)>(
                hops::MarkovChainType::HitAndRun,
                Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>(A * roundingTransformation),
                b,
                startingPoint,
                roundingTransformation,
                decltype(startingPoint)::Zero(roundingTransformation.rows()),
                model,
                false);
    } else {
        std::cerr << "No chain with chainname " << chainName << std::endl;
        std::exit(1);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    float upperLimitAcceptanceRate = chainName == "CSmMALA" ? 0.65 : 0.3;
    float lowerLimitAcceptanceRate = chainName == "CSmMALA" ? 0.3 : 0.20;
    double lowerLimitStepSize = 1e-10;
    double upperLimitStepSize = (chainName == "HRR" || chainName == "CHRR") ? 1e6 : 1;
    size_t iterationsToTestStepSize = 100 * A.cols();
    size_t maxIterations = 10000 * A.cols();

    bool isTuned = hops::AcceptanceRateTuner::tune(markovChain.get(),
                                                   randomNumberGenerator,
                                                   {lowerLimitAcceptanceRate,
                                                    upperLimitAcceptanceRate,
                                                    lowerLimitStepSize,
                                                    upperLimitStepSize,
                                                    iterationsToTestStepSize,
                                                    maxIterations});
    std::cout << "isTuned: " << isTuned << std::endl;
    std::cout << "current step size: " << markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE) << std::endl;

    auto fileWriter = hops::FileWriterFactory::createFileWriter(modelName + "_" + markovChain->getName(),
                                                                hops::FileWriterType::Csv);
    long thinning = A.cols() * 10;
    long numberOfSamples = 100;
    for (int i = 0; i < 1000; ++i) {
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
        markovChain->writeHistory(fileWriter.get());
        markovChain->clearHistory();
    }
}

