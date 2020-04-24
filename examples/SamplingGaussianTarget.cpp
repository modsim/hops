#include <Eigen/SparseCore>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/MarkovChain/MarkovChainFactory.hpp>
#include <hops/MarkovChain/Recorder/StateRecorder.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/ChordStepDistributions.hpp>
#include <hops/MarkovChain/Proposal/DikinProposal.hpp>
#include <hops/Model/ModelMixin.hpp>
#include <hops/Model/MultivariateGaussianModel.hpp>
#include <hops/MarkovChain/Proposal/CSmMALAProposal.hpp>
#include <hops/MarkovChain/Proposal/HitAndRunProposal.hpp>
#include <utility>

template<typename MarkovChainProposer>
std::unique_ptr<hops::MarkovChain>
wrapMarkovChainProposer(const MarkovChainProposer &markovChainProposer,
                        double stepSize) {
    auto mc = hops::MarkovChainAdapter(
            hops::TimestampRecorder(
                    hops::StateRecorder(
                            hops::MetropolisHastingsFilter(
                                    markovChainProposer
                            )
                    )
            )
    );
    mc.setStepSize(stepSize);
    return std::make_unique<decltype(mc)>(mc);
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        std::cout << "arg " << argv[i] << std::endl;
    }
    std::string Afile(argv[1]);
    std::string bfile(argv[2]);
    std::string sfile(argv[3]);
    std::string Nfile(argv[4]);
    std::string Tfile(argv[5]);
    std::string shiftfile(argv[6]);
    std::string meanfile(argv[7]);
    std::string chainName(argv[8]);
    double stepSize = std::stod(argv[9]);

    auto A = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(Afile);
    auto b = hops::CsvReader::readVector<Eigen::VectorXd>(bfile);
    auto s = hops::CsvReader::readVector<Eigen::VectorXd>(sfile);
    auto N = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(Nfile);
    auto T = hops::CsvReader::readMatrix<Eigen::SparseMatrix<double>>(Tfile);
    auto shift = hops::CsvReader::readVector<Eigen::VectorXd>(shiftfile);
    auto mean = hops::CsvReader::readVector<Eigen::VectorXd>(meanfile);

    Eigen::MatrixXd covariance = 0.01 * mean.asDiagonal();

    std::unique_ptr<hops::MarkovChain> markovChain;
    if (chainName == "DikinWalk") {
        auto proposer = hops::ModelMixin(hops::DikinProposal(A, b, s),
                                         hops::MultivariateGaussianModel(mean, covariance));
        markovChain = wrapMarkovChainProposer(proposer, stepSize);
    } else if (chainName == "CSmMALA") {
        markovChain = wrapMarkovChainProposer(
                hops::CSmMALAProposal(hops::MultivariateGaussianModel(mean, covariance), A, b, s),
                stepSize);
    } else if (chainName == "CHRR") {
        auto proposer = hops::ModelMixin(
                hops::StateTransformation(
                        hops::CoordinateHitAndRunProposal<Eigen::MatrixXd, Eigen::VectorXd, hops::GaussianStepDistribution<double>>(
                                A, b, s),
                        hops::Transformation<Eigen::MatrixXd, Eigen::VectorXd>(N, shift)),
                hops::MultivariateGaussianModel(mean, covariance)
        );
        markovChain = wrapMarkovChainProposer(proposer, stepSize);
    } else if (chainName == "HRR") {
        auto proposer = hops::ModelMixin(
                hops::StateTransformation(
                        hops::HitAndRunProposal<Eigen::MatrixXd, Eigen::VectorXd, hops::GaussianStepDistribution<double>>(
                                A, b, s),
                        hops::Transformation<Eigen::MatrixXd, Eigen::VectorXd>(N, shift)),
                hops::MultivariateGaussianModel(mean, covariance)
        );
        markovChain = wrapMarkovChainProposer(proposer, stepSize);
    } else {
        std::cerr << "No chain with chainname " << chainName << std::endl;
        std::exit(1);
    }

    auto fileWriter = hops::FileWriterFactory::createFileWriter(markovChain->getName(), hops::FileWriterType::Csv);
    hops::RandomNumberGenerator randomNumberGenerator((std::random_device()()));
    long thinning = mean.rows() * 100;
    long numberOfSamples = 1000;
    for (int i = 0; i < 50; ++i) {
        markovChain->draw(randomNumberGenerator, numberOfSamples, thinning);
        markovChain->writeHistory(fileWriter.get());
        markovChain->clearHistory();
    }
}

