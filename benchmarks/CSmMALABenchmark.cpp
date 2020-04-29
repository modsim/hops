#include <Eigen/SparseCore>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/LinearProgram/LinearProgramFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/AcceptanceRateTuner.hpp>
#include <hops/PolytopePreprocessing/NormalizePolytope.hpp>

int main(int argc, char **argv) {
    std::string modelDirectory("../resources/");
    std::string modelName("iAT_PLT_636");
    std::string chainName("CSmMALA");

    // TODO fix paths for for windows
    std::string Afile = modelDirectory + "/" + modelName + "/A_" + modelName + "_unrounded.csv";
    std::string bfile = modelDirectory + "/" + modelName + "/b_" + modelName + "_unrounded.csv";
    std::string meanFile = modelDirectory + "/" + modelName + "/start_" + modelName + "_unrounded.csv";

    auto A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(Afile);
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>(bfile);
    auto mean = hops::CsvReader::readVector<Eigen::VectorXd>(meanFile);
    Eigen::MatrixXd denseA = A;
    hops::normalizePolytope(denseA, b);
    A = denseA.sparseView();
    A.makeCompressed();

    Eigen::MatrixXd covariance = 1e-6 * Eigen::VectorXd::Ones(mean.rows()).asDiagonal();
    hops::MultivariateGaussianModel model(mean, covariance);

    std::unique_ptr<hops::LinearProgram> linearProgram = hops::LinearProgramFactory::createLinearProgram(
            Eigen::MatrixXd(A),
            b);
    Eigen::VectorXd startingPoint = linearProgram->calculateChebyshevCenter().optimalParameters;

    std::unique_ptr<hops::MarkovChain> sparseCSmMALA = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::CSmMALA,
            A,
            b,
            startingPoint,
            model,
            false);
    std::unique_ptr<hops::MarkovChain> denseCSmMALA = hops::MarkovChainFactory::createMarkovChain(
            hops::MarkovChainType::CSmMALA,
            Eigen::MatrixXd(A),
            b,
            startingPoint,
            model,
            false);

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    float upperLimitAcceptanceRate = chainName == "CSmMALA" ? 0.65 : 0.3;
    float lowerLimitAcceptanceRate = chainName == "CSmMALA" ? 0.3 : 0.20;
    double lowerLimitStepSize = 1e-10;
    double upperLimitStepSize = 1;
    size_t iterationsToTestStepSize = 10 * A.cols();
    size_t maxIterations = 1000 * A.cols();

//    bool isTuned = hops::AcceptanceRateTuner::tune(sparseCSmMALA.get(),
//                                                   randomNumberGenerator,
//                                                   {lowerLimitAcceptanceRate,
//                                                    upperLimitAcceptanceRate,
//                                                    lowerLimitStepSize,
//                                                    upperLimitStepSize,
//                                                    iterationsToTestStepSize,
//                                                    maxIterations});
//    std::cout << "isTuned: " << isTuned << std::endl;
//    std::cout << "current step size: " << sparseCSmMALA->getAttribute(hops::MarkovChainAttribute::STEP_SIZE)
//              << std::endl;

    denseCSmMALA->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);
    sparseCSmMALA->setAttribute(hops::MarkovChainAttribute::STEP_SIZE, 1);

    long thinning = A.cols() * 10;
    long numberOfSamples = 10;

    long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    sparseCSmMALA->draw(randomNumberGenerator, numberOfSamples, thinning);
    long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << "Sparse sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / (1000 * thinning * numberOfSamples) << " s per sample"
              << std::endl;

    startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    denseCSmMALA->draw(randomNumberGenerator, numberOfSamples, thinning);
    endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
    std::cout << "Dense sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
              << " seconds, that's "
              << static_cast<double>(endEpoch - startEpoch) / (1000 * thinning * numberOfSamples) << " s per sample"
              << std::endl;
}

