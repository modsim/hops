#include <any>
#include <iostream>
#include <iomanip>

#include "hops/hops.hpp"


std::tuple<Eigen::MatrixXd, Eigen::VectorXd> createSimplex(size_t dims) {
    Eigen::MatrixXd A(dims + 1, dims);
    Eigen::VectorXd upperBounds(dims);
    for (size_t i = 0; i < dims; ++i) {
        upperBounds(i) = 1.;
    }
    A << upperBounds.transpose(), -Eigen::MatrixXd::Identity(dims, dims);

    Eigen::VectorXd b(dims + 1);
    b << 5, Eigen::VectorXd::Zero(dims);

    return std::make_tuple(A, b);
}

int main() {
    std::srand(42);
    const long numberOfSamples = 2'00;

    double covScale = 1e-2;
    const size_t dimStep = 2;
    Eigen::MatrixXd baseCov = covScale * Eigen::MatrixXd::Random(dimStep, dimStep);
    const Eigen::MatrixXd baseCovariance =
            baseCov.transpose() * baseCov + covScale * Eigen::MatrixXd::Identity(dimStep, dimStep);;

    std::vector<
            std::tuple<
                    Eigen::MatrixXd,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    hops::Gaussian,
                    hops::MarkovChainType,
                    long,
                    long,
                    double,
                    double,
                    bool,
                    std::string>
    > runConfigurations;

    size_t dims = 10;
    Eigen::MatrixXd A(2 * dims, dims);
    A << Eigen::MatrixXd::Identity(dims, dims), -Eigen::MatrixXd::Identity(dims, dims);

    Eigen::VectorXd b(2 * dims);
    b << 5 * Eigen::VectorXd::Ones(dims), Eigen::VectorXd::Zero(dims);


    Eigen::VectorXd mean = Eigen::VectorXd::Zero(dims);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dims, dims);
    for (size_t i = 0; i < dims; i += dimStep) {
        covariance.block(i, i, dimStep, dimStep) = baseCovariance;
    }

    hops::Gaussian model(mean, covariance);
    mean = Eigen::VectorXd::Ones(dims);

    auto gibbs = hops::MarkovChainAdapter(
            hops::MetropolisHastingsFilter(
                    hops::TruncatedGaussianProposal(
                            A,
                            b,
                            mean,
                            model
                    )
            )
    );

    auto chrr = hops::MarkovChainFactory::createMarkovChain(hops::MarkovChainType::CoordinateHitAndRun,
                                                            A, b, mean, model);
    auto bmala = hops::MarkovChainFactory::createMarkovChain(hops::MarkovChainType::BilliardMALA,
                                                             A, b, mean, model);

    hops::RandomNumberGenerator rng(42);
    std::vector<Eigen::VectorXd> samples;
    for (long n = 0; n < numberOfSamples; ++n) {
        auto state = gibbs.draw(rng).second;
        samples.emplace_back(state);
    }
    auto gibbs_writer = hops::FileWriterFactory::createFileWriter("tvn2_gibbs", hops::FileWriterType::CSV);
    gibbs_writer->write("states", samples);


    std::vector<Eigen::VectorXd> chrr_samples;
    for (long n = 0; n < numberOfSamples; ++n) {
        auto state = chrr->draw(rng).second;
        chrr_samples.emplace_back(state);
    }
    auto chrr_writer = hops::FileWriterFactory::createFileWriter("chrr", hops::FileWriterType::CSV);
    chrr_writer->write("states", chrr_samples);

    std::vector<Eigen::VectorXd> bmala_samples;
    for (long n = 0; n < numberOfSamples; ++n) {
        auto state = bmala->draw(rng).second;
        bmala_samples.emplace_back(state);
    }
    auto bmala_writer = hops::FileWriterFactory::createFileWriter("bmala", hops::FileWriterType::CSV);
    bmala_writer->write("states", bmala_samples);
}
