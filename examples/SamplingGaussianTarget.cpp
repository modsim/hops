#include <Eigen/SparseCore>
#include <iostream>
#include <filesystem>
#include <hops/hops.hpp>

int main(int argc, char **argv) {
    if (argc != 8 && argc != 7) {
        std::cout << "usage: SamplingGaussianTarget A.csv b.csv mean.csv covariance.csv "
                  << "numberOfSamples thinningNumber CHRR|HRR|DikinWalk outputName [startingPoint.csv]"
                  << "\nArgument Description:\n"
                  << "\tinput-path\t\t path with model\n"
                  << "\tnumberOfSamples\t\t number of samples to generate\n"
                  << "\tthinningNumber\t\t number of markov chain iterations per sample\n"
                  << "\talgorithm\t\t\t CHRR or HRR or DikinWalk\n"
                  << "\toutputName\t\t\t name for output\n"
                  << "\tfisherweight\t\t\t fisherweight\n" << std::endl;
        exit(0);
    }

    std::filesystem::path inputPath = std::string(argv[1]);
    std::string modelName = inputPath.filename();

    std::string Afile = inputPath / std::filesystem::path("A_" + modelName + "_unrounded.csv");
    std::string bfile = inputPath / std::filesystem::path("b_" + modelName + "_unrounded.csv");
    std::string meanFile =
            inputPath / std::filesystem::path("MeanVector.csv");
    std::string covFile =
            inputPath / std::filesystem::path("CovarianceMatrix.csv");

    Eigen::SparseMatrix<double> A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(Afile);
    Eigen::VectorXd b = hops::CsvReader::readVector<Eigen::VectorXd>(bfile);
    Eigen::VectorXd mean = hops::CsvReader::readVector<Eigen::VectorXd>(meanFile);
    Eigen::MatrixXd covariance = hops::CsvReader::readMatrix<Eigen::MatrixXd>(covFile, true);

    decltype(mean) startingPoint = mean;

    long numberOfSamples = std::strtol(argv[2], NULL, 10);
    long thinning = std::strtol(argv[3], NULL, 10);
    std::string chainName = argv[4];

    hops::MultivariateGaussianModel model(mean, covariance);

    std::shared_ptr<hops::MarkovChain> markovChain;

    if (chainName == "DikinWalk") {
        hops::MarkovChainType chainType = hops::MarkovChainType::DikinWalk;
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
    } else if (chainName == "CSmMALA") {
        hops::MarkovChainType chainType = hops::MarkovChainType::CSmMALA;
        markovChain = hops::MarkovChainFactory::createMarkovChain(chainType,
                                                                  A,
                                                                  b,
                                                                  startingPoint,
                                                                  model);

        double fisherWeight = std::stod(argv[6]);
        std::cout << "setting fisherweight to " << fisherWeight << std::endl;
        markovChain->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, fisherWeight);
    } else if (chainName == "CHRR" || chainName == "HRR") {
        hops::MarkovChainType chainType =
                chainName == "CHRR" ? hops::MarkovChainType::CoordinateHitAndRun : hops::MarkovChainType::HitAndRun;

        std::string Afile_rounded = inputPath / std::filesystem::path("A_" + modelName + "_rounded.csv");
        std::string bfile_rounded = inputPath / std::filesystem::path("b_" + modelName + "_rounded.csv");
        std::string transformFile =
                inputPath / std::filesystem::path("T_" + modelName + "_rounded.csv");
        std::string shiftFile =
                inputPath / std::filesystem::path("shift_" + modelName + "_rounded.csv");

        // TODO round
        std::string startFile =
                inputPath / std::filesystem::path("start_" + modelName + "_rounded.csv");

        Eigen::MatrixXd A_rounded = hops::CsvReader::readMatrix<decltype(A)>(Afile_rounded);
        auto b_rounded = hops::CsvReader::readVector<decltype(b)>(bfile_rounded);
        Eigen::MatrixXd transformation_rounded = hops::CsvReader::readMatrix<decltype(A)>(transformFile);
        auto shift_rounded = hops::CsvReader::readVector<decltype(b)>(shiftFile);

        Eigen::VectorXd startingPoint_rounded;
        if(transformation_rounded.isLowerTriangular()) {
            startingPoint_rounded = transformation_rounded.template triangularView<Eigen::Lower>().solve(
                    mean);
        }
        else if (transformation_rounded.isUpperTriangular()) {
            startingPoint_rounded = transformation_rounded.template triangularView<Eigen::Upper>().solve(
                    mean);
        }
        else {
            throw std::runtime_error("rounding transformation is not upper or lower triangular. It can't be correct.");
        }


        markovChain = hops::MarkovChainFactory::createMarkovChain<Eigen::MatrixXd, decltype(b), decltype(model)>(
                chainType,
                A_rounded,
                b_rounded,
                startingPoint_rounded,
                transformation_rounded,
                shift_rounded,
                model);
    } else {
        std::cerr << "No chain with chainname " << chainName << std::endl;
        std::exit(1);
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    double lowerLimitStepSize = 1e-15;
    double upperLimitStepSize = 1;

    size_t iterationsToTestStepSize = 200;
    size_t posteriorUpdateIterations = 600;
    size_t pureSamplingIterations = 15;
    size_t stepSizeGridSize = 500;
    size_t iterationsForConvergence = 25;
    double smoothingLength = 0.002;
    bool recordData = true;

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

    auto fileWriter = hops::FileWriterFactory::createFileWriter(inputPath.string() + "_" +markovChain->getName() + "_" + std::string(argv[5]),
                                                                hops::FileWriterType::CSV);
    markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}
