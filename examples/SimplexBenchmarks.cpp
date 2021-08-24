#include <hops/hops.hpp>
#include <Eigen/SparseCore>
#include <exception>
#include <utility>
#include <random>
#include <array>
#include <execution>

void print(std::vector<double> vec) {
    for (auto &v: vec) {
        std::cout << v << "\t";
    }
    std::cout << std::endl;
}

std::vector<double> createDimLengths(int n, double smallest, double largest) {
    if (n < 0) {
        throw std::runtime_error("n<0 is not possible");
    }
    double interval_length = largest - smallest;
    std::vector<double> dimLengths;
    dimLengths.reserve(n);
    for (int i = 0; i < n; ++i) {
        dimLengths.emplace_back(
                smallest + interval_length * i / (n - 1)
        );
    }
    return dimLengths;
}

std::tuple<Eigen::MatrixXd, Eigen::VectorXd> createSimplex(std::vector<double> dimLengths) {
    Eigen::MatrixXd A(dimLengths.size() + 1, dimLengths.size());
    Eigen::VectorXd upperBounds(dimLengths.size());
    for (size_t i = 0; i < dimLengths.size(); ++i) {
        upperBounds(i) = 1. / dimLengths[i];
    }
    A << upperBounds.transpose(), -Eigen::MatrixXd::Identity(dimLengths.size(), dimLengths.size());

    Eigen::VectorXd b(dimLengths.size() + 1);
    b << 1, Eigen::VectorXd::Zero(dimLengths.size());

//    std::cout << "A " << std::endl << A << std::endl;
//    std::cout << "b " << std::endl << b << std::endl;
    return std::make_tuple(A, b);
}

hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>
createGaussian(const std::vector<double> &dimLengths) {
    double offDiagonalScaling = 0.2;
    long n = dimLengths.size();
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(n, n);

    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            if (i == j) {
                cov(i, i) = std::pow(1. / dimLengths[i], 2);
                cov(j, j) = std::pow(1. / dimLengths[j], 2);
            }
        }
    }
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            if (i != j) {
                cov(i, j) = offDiagonalScaling * std::sqrt(cov(i, i) * cov(j, j));
            }
        }
    }

    return hops::MultivariateGaussianModel(std::move(mu), std::move(cov));
}

void run(hops::MarkovChainType markovChainType, const std::string& unique_id, int n, double smallest, double largest, long numSamples = 1e4,
         long thinningFactor = 1e2, double targetAcceptanceRate=0.23, const std::string& fisherWeight="0") {
    auto dimLengths = createDimLengths(n, smallest, largest);
//    print(dimLengths);

    auto[A, b] = createSimplex(dimLengths);
    auto gaussian = createGaussian(dimLengths);

    hops::normalizePolytope(A, b);

    std::shared_ptr<hops::MarkovChain> markovChain;
    if (markovChainType == hops::MarkovChainType::CoordinateHitAndRun) {
        auto MVE = hops::MaximumVolumeEllipsoid<double>::construct(A, b, 1e6);
        std::cout << std::boolalpha << "MVE converged: " << MVE.hasConverged() << std::endl;

        Eigen::MatrixXd roundedA = A * MVE.getRoundingTransformation();

        auto linearProgram = hops::LinearProgramFactory::createLinearProgram(roundedA, b);
        auto roundedStart = linearProgram->computeChebyshevCenter().optimalParameters;

        markovChain = hops::MarkovChainFactory::createMarkovChain<decltype(A), decltype(b)>(
                markovChainType,
                roundedA,
                b,
                roundedStart,
                Eigen::MatrixXd(MVE.getRoundingTransformation()),
                Eigen::VectorXd::Zero(roundedStart.rows()),
                gaussian);


    } else {
        auto linearProgram = hops::LinearProgramFactory::createLinearProgram(A, b);
        auto start = linearProgram->computeChebyshevCenter().optimalParameters;

        Eigen::SparseMatrix<double> sparseA = A.sparseView();

        markovChain = hops::MarkovChainFactory::createMarkovChain<decltype(sparseA), decltype(b)>(
                markovChainType,
                sparseA,
                b,
                start,
                gaussian);

        std::cout << fisherWeight << " " << std::stod(fisherWeight) << std::endl;
        if(markovChainType==hops::MarkovChainType::CSmMALA) {
            markovChain->setAttribute(hops::MarkovChainAttribute::FISHER_WEIGHT, std::stod(fisherWeight));
        }
    }

    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
    for(int i=0; i<3; ++i) {
        markovChain->draw(randomNumberGenerator, 1000);
        try {
            double lowerLimitStepSize = 1e-10;
            double upperLimitStepSize = 1;

            size_t iterationsToTestStepSize = 100;
            size_t posteriorUpdateIterations = 50;
            size_t pureSamplingIterations = 10;
            size_t stepSizeGridSize = 60;
            size_t iterationsForConvergence = 3;
            double smoothingLength = lowerLimitStepSize;
            bool recordData = true;


            hops::AcceptanceRateTuner::param_type tuningParameters(targetAcceptanceRate,
                                                                   iterationsToTestStepSize,
                                                                   posteriorUpdateIterations,
                                                                   pureSamplingIterations,
                                                                   iterationsForConvergence,
                                                                   stepSizeGridSize,
                                                                   lowerLimitStepSize,
                                                                   upperLimitStepSize,
                                                                   smoothingLength,
                                                                   long(std::random_device()()),
                                                                   recordData);

            std::vector<decltype(markovChain)> tuningChains = {markovChain};
            std::vector<decltype(randomNumberGenerator)> randomNumberGenerators = {randomNumberGenerator};
            hops::AcceptanceRateTuner::tune(tuningChains,
                                            randomNumberGenerators,
                                            tuningParameters);
        }
        catch (std::runtime_error &e) {
            std::cout << "Skipping tuning (Reason: " << e.what() << ")" << std::endl;
        }
    }

    std::string markovChainName = hops::MarkovChainTypeToShortcutString(markovChainType);
    std::string fileWriterOutput = markovChainName + (markovChainName == "CSmMALA" ? fisherWeight : "") + "_acceptance" + std::to_string(targetAcceptanceRate) + "_" + unique_id;
    auto fileWriter = hops::FileWriterFactory::createFileWriter(fileWriterOutput,
                                                                hops::FileWriterType::CSV);
    Eigen::VectorXd stepSize(1);
    stepSize(0) = markovChain->getAttribute(hops::MarkovChainAttribute::STEP_SIZE);
    fileWriter->write("stepSize", stepSize);


    markovChain->draw(randomNumberGenerator, numSamples, thinningFactor * n);
    markovChain->writeHistory(fileWriter.get());
    markovChain->clearHistory();
}

int main() {
    std::array<hops::MarkovChainType, 6> types = {
            hops::MarkovChainType::CoordinateHitAndRun,
            hops::MarkovChainType::CSmMALA,
            hops::MarkovChainType::CSmMALA,
            hops::MarkovChainType::CSmMALA,
            hops::MarkovChainType::DikinWalk,
            hops::MarkovChainType::HitAndRun,
            };
    std::array<double, 6> acceptanceRates = { 0.23, 0.23, 0.23, 0.5, 0.23, 0.23 };
    std::array<std::string, 6> fisherWeights = {"0", "0", "0.5", "0.5", "0", "0"};

    struct Config {
        hops::MarkovChainType markovChainType;
        std::string unique_id;
        int n;
        double smallest;
        double longest;
        long numSamples;
        long thinning;
        double targetAcceptanceRate;
        std::string fisherWeight;
    };

    int n = 3;
    double smallest = 1e-2;
    double largest = 1e2;
    long numSamples = 1e4;
    long thinning = 100;

    std::vector<Config> configs;
    for(int i = 0; i<types.size(); ++i) {
        Config config;
        config.markovChainType = types[i];
        config.targetAcceptanceRate = acceptanceRates[i];
        config.fisherWeight = fisherWeights[i];
        config.n = n;
        config.smallest = smallest;
        config.longest = largest;
        config.unique_id = std::to_string(n) + "d";
        config.numSamples = numSamples;
        config.thinning = thinning;

        configs.emplace_back(config);
    }


    for(int dim=5; dim<20; dim+=5) {
        for (int i = 0; i < types.size(); ++i) {
            Config config;
            config.markovChainType = types[i];
            config.targetAcceptanceRate = acceptanceRates[i];
            config.fisherWeight = fisherWeights[i];
            config.n = dim;
            config.smallest = smallest;
            config.longest = largest;
            config.unique_id = std::to_string(dim) + "d";
            config.numSamples = numSamples;
            config.thinning = thinning;

            configs.emplace_back(config);
        }
    }


    std::for_each(std::execution::par, configs.begin(),configs.end(),
                  [&](Config& c){
        run(c.markovChainType, c.unique_id, c.n,  c.smallest, c.longest, c.numSamples, c.thinning, c.targetAcceptanceRate, c.fisherWeight);
    } );
}


