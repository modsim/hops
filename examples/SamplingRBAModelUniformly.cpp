#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/Model/ModelMixin.hpp>

int main(int argc, char **argv) {
    assert(argc == 7);
    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));

    std::cout
            << "program: " << std::string(argv[0]) << std::endl
            << "A: " << std::string(argv[1]) << std::endl
            << "b: " << std::string(argv[2]) << std::endl
            << "mu: " << std::string(argv[3]) << std::endl
            << "cov: " << std::string(argv[4]) << std::endl
            << "outdir: " << std::string(argv[5]) << std::endl
            << "rng: " << std::string(argv[6]) << std::endl;
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

    auto fileWriter1 = hops::FileWriterFactory::createFileWriter(std::string(argv[5]) + "uchr",
                                                                 hops::FileWriterType::Csv);
    auto markovChain1 = hops::MarkovChainAdapter(
            hops::NoOpDrawAdapter(
                    hops::TimestampRecorder(
                            hops::StateRecorder(
                                    hops::CoordinateHitAndRunProposal(
                                            A,
                                            b,
                                            mean)
                            )
                    )
            )
    );

    long thinning = A.cols() * 1000;
    long numberOfSamples = 10000;
    while (true) {
        long startEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        markovChain1.draw(randomNumberGenerator, numberOfSamples, thinning);
        long endEpoch = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
        ).count();
        std::cout << "Sampling took " << static_cast<double>(endEpoch - startEpoch) / 1000
                  << " seconds, that's "
                  << static_cast<double>(endEpoch - startEpoch) / static_cast<double>(numberOfSamples * thinning * 1000)
                  << " s per sample"
                  << std::endl;
        markovChain1.writeHistory(fileWriter1.get());
        markovChain1.clearHistory();
    }
}

