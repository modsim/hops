#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/LinearProgram/LinearProgramGurobiImpl.hpp>

int main(int argc, char **argv) {
    assert(argc == 6);
    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    std::cout
            << std::string(argv[0]) << std::endl
            << std::string(argv[1]) << std::endl
            << std::string(argv[2]) << std::endl
            << std::string(argv[3]) << std::endl
            << std::string(argv[4]) << std::endl
            << std::string(argv[5]) << std::endl;
    Eigen::MatrixXd A;
    Eigen::MatrixXd covariance;
    try {
        A = hops::CsvReader::readMatrix<Eigen::MatrixXd>(std::string(argv[1]));
    }
    catch (const std::invalid_argument &) {
        A = hops::CsvReader::readMatrix<Eigen::MatrixXd>(std::string(argv[1]), true);
    }
    try {
        covariance = hops::CsvReader::readMatrix<Eigen::MatrixXd>(std::string(argv[4]));
    }
    catch (const std::invalid_argument &) {
        covariance = hops::CsvReader::readMatrix<Eigen::MatrixXd>(std::string(argv[4]), true);
    }

    auto b = hops::CsvReader::readVector<Eigen::VectorXd>(std::string(argv[2]));
    auto mean = hops::CsvReader::readVector<Eigen::VectorXd>(std::string(argv[3]));

    auto fileWriter1 = hops::FileWriterFactory::createFileWriter(std::string(argv[5]) + "dikin",
                                                                 hops::FileWriterType::Csv);
    auto fileWriter2 = hops::FileWriterFactory::createFileWriter(std::string(argv[5]) + "csmmala",
                                                                 hops::FileWriterType::Csv);


    hops::LinearProgramGurobiImpl linearProgramGurobi(A, b);
    auto solution = linearProgramGurobi.calculateChebyshevCenter();
    Eigen::VectorXd s = solution.optimalParameters;
    assert(((A * s - b).array() < 0).all());
    auto markovChain1 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::ModelMixin(
                                    hops::DikinProposal(
                                            A, b, s
                                    ),
                                    hops::MultivariateGaussianModel(mean, covariance)
                            )
                    )
            )
    );
    markovChain1.setStepSize(3. / 40);

    auto markovChain2 = hops::MarkovChainAdapter(
            hops::StateRecorder(
                    hops::MetropolisHastingsFilter(
                            hops::CSmMALAProposal(
                                    hops::MultivariateGaussianModel(mean, covariance), A, b, s)
                    )
            )
    );
    markovChain2.setStepSize(3. / 40);

    long thinning = 1; // TODO mean.rows() * 100;
    long numberOfSamples = 1000;
    while (true) {
        long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        markovChain1.draw(randomNumberGenerator, numberOfSamples, thinning);
        markovChain2.draw(randomNumberGenerator, numberOfSamples, thinning);
        long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
                  << " seconds, that's "
                  << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
                  << " s per sample"
                  << std::endl;
        markovChain1.writeHistory(fileWriter1.get());
        markovChain2.writeHistory(fileWriter2.get());
        markovChain1.clearHistory();
        markovChain2.clearHistory();
    }
}

