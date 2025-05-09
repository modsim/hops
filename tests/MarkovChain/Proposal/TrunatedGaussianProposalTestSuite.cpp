#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE TruncatedGaussianProposalTestSuite

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <Eigen/Core>

#include "hops/FileReader/CsvReader.hpp"
#include "hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp"
#include "hops/MarkovChain/MarkovChainAdapter.hpp"
#include "hops/MarkovChain/ModelWrapper.hpp"
#include "hops/MarkovChain/Proposal/TruncatedGaussianProposal.hpp"
#include "hops/Model/Gaussian.hpp"
#include "hops/RandomNumberGenerator/RandomNumberGenerator.hpp"
#include "hops/Statistics/EffectiveSampleSize.hpp"

BOOST_AUTO_TEST_SUITE(TruncatedGaussianProposal)

    BOOST_AUTO_TEST_CASE(DimensionNames) {
        const long rows = 2;
        const long cols = 1;
        Eigen::MatrixXd A(rows, cols);
        A << 1, -1;
        Eigen::VectorXd b = 1000 * Eigen::VectorXd::Ones(rows);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = std::make_shared<hops::Gaussian>(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));
        auto gaussian = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));

        auto proposer = hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        gaussian);

        std::vector<std::string> expectedNames = {"x_0"};
        auto actualNames = proposer.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }

        expectedNames = std::vector<std::string>{"y_1"};
        proposer.setDimensionNames(expectedNames);
        actualNames = proposer.getDimensionNames();

        BOOST_CHECK_EQUAL(actualNames.size(), expectedNames.size());
        for (size_t i = 0; i < expectedNames.size(); ++i) {
            BOOST_CHECK_EQUAL(actualNames[i], expectedNames[i]);
        }

    }

    BOOST_AUTO_TEST_CASE(GaussianInCube) {
        const long rows = 6;
        const long cols = 3;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                -1, 0, 0,
                0, -1, 0,
                0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));


        hops::TruncatedGaussianProposal proposer(A,
                                                 b,
                                                 interiorPoint,
                                                 model);

        hops::RandomNumberGenerator randomNumberGenerator(42);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() >= 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;

        BOOST_CHECK(proposer.getModel() != nullptr);
    }


    BOOST_AUTO_TEST_CASE(GaussianInWideCubeHasCorrectStd) {
        const long rows = 2;
        const long cols = 1;
        Eigen::MatrixXd A(rows, cols);
        A << 1, -1;
        Eigen::VectorXd b = 1000 * Eigen::VectorXd::Ones(rows);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = std::make_shared<hops::Gaussian>(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));
        auto gaussian = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        gaussian)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        std::vector<Eigen::VectorXd> samples;
        double num_samples = 50'000;
        Eigen::VectorXd sample_sum = Eigen::VectorXd::Zero(1);
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            samples.emplace_back(state);
            sample_sum += state;
        }

        double mean = sample_sum(0) / samples.size();


        double ess = hops::computeEffectiveSampleSize(std::vector<decltype(samples)>{samples}, 0);
        double standardErrorOfMean = 1. / std::sqrt(ess);
        double sq_sum = 0;
        for (const auto &s:samples) {
            sq_sum += (s(0) - mean) * (s(0) - mean);
        }
        double stdev = std::sqrt(sq_sum / (samples.size() - 1));

        BOOST_CHECK(std::abs(mean - model->getMean()(0)) < 2 * standardErrorOfMean);
        BOOST_CHECK_CLOSE(stdev, 1., 1);
    }

    BOOST_AUTO_TEST_CASE(OtherGaussianInWideCubeHasCorrectStd) {
        const long rows = 2;
        const long cols = 1;
        Eigen::MatrixXd A(rows, cols);
        A << 1, -1;
        Eigen::VectorXd b = 1000 * Eigen::VectorXd::Ones(rows);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 5;
        }

        auto model = std::make_shared<hops::Gaussian>(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));
        auto gaussian = hops::Gaussian(interiorPoint, 0.3 * Eigen::MatrixXd::Identity(cols, cols));

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        gaussian)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        std::vector<Eigen::VectorXd> samples;
        double num_samples = 50'000;
        Eigen::VectorXd sample_sum = Eigen::VectorXd::Zero(1);
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            samples.emplace_back(state);
            sample_sum += state;
        }

        double mean = sample_sum(0) / samples.size();


        double ess = hops::computeEffectiveSampleSize(std::vector<decltype(samples)>{samples}, 0);
        double standardErrorOfMean = 1. / std::sqrt(ess);
        double sq_sum = 0;
        for (const auto &s:samples) {
            sq_sum += (s(0) - mean) * (s(0) - mean);
        }
        double stdev = std::sqrt(sq_sum / (samples.size() - 1));

        BOOST_CHECK(std::abs(mean - model->getMean()(0)) < 2 * standardErrorOfMean);
        BOOST_CHECK_CLOSE(stdev, std::sqrt(0.3), 1);
    }

    BOOST_AUTO_TEST_CASE(Other2DGaussianInWideCubeHasCorrectStd) {
        const long rows = 4;
        const long cols = 2;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0, 1, -1, 0, 0, -1;
        Eigen::VectorXd b = 1000 * Eigen::VectorXd::Ones(rows);
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 5;
        }

        auto model = std::make_shared<hops::Gaussian>(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));
        double expected_std = 0.5;
        auto gaussian = hops::Gaussian(interiorPoint,
                                       std::pow(expected_std, 2) * Eigen::MatrixXd::Identity(cols, cols));

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        gaussian)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        std::vector<Eigen::VectorXd> samples;
        double num_samples = 50'000;
        Eigen::VectorXd sample_sum = Eigen::VectorXd::Zero(cols);
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            samples.emplace_back(state);
            sample_sum += state;
        }

        long col = 0;
        double mean = sample_sum(col) / samples.size();

        double ess = hops::computeEffectiveSampleSize(std::vector<decltype(samples)>{samples}, col);
        double standardErrorOfMean = expected_std / std::sqrt(ess);
        double sq_sum = 0;
        for (const auto &s:samples) {
            sq_sum += (s(col) - mean) * (s(col) - mean);
        }
        double stdev = std::sqrt(sq_sum / (samples.size() - 1));

        BOOST_CHECK(std::abs(mean - model->getMean()(col)) < 2 * standardErrorOfMean);
        BOOST_CHECK_CLOSE(stdev, expected_std, 1);
    }

    BOOST_AUTO_TEST_CASE(AntonsProblemDoesNotProduceNaNs) {
        const long rows = 8;
        const long cols = 2;
        Eigen::MatrixXd A(rows, cols);
        A << -1., 0., 0., -1., 1., 0., 0., 1., 1., -1., 1., -1., 1., -1., 1., -1.;

        Eigen::VectorXd b(rows);
        b << 0., 0., 5., 5., 0., 0., 0., -0.;

        Eigen::VectorXd interiorPoint(cols);
        interiorPoint << 0.6, 1.0;
        Eigen::MatrixXd cov(cols, cols);
        cov << 0.0001, 0., 0., 0.0001;

        hops::Gaussian model(interiorPoint, cov);

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        model)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        double num_samples = 10'000;
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            BOOST_CHECK(std::isfinite(state(0)));
            BOOST_CHECK(std::isfinite(state(1)));
        }
    }

    BOOST_AUTO_TEST_CASE(AntonsProblemDoesNotProduceNaNs2) {
        const long rows = 8;
        const long cols = 2;
        Eigen::MatrixXd A(rows, cols);
        A << -1., 0., 0., -1., 1., 0., 0., 1., 1., -1., 1., -1., 1., -1., 1., -1.;

        Eigen::VectorXd b(rows);
        b << 0., 0., 5., 5., 0., 0., 0., -0.;

        Eigen::VectorXd mean(cols);
        mean << 0.6, 1.0;
        Eigen::MatrixXd cov(cols, cols);
        cov << 0.0001, 0., 0., 0.0001;

        hops::Gaussian model(mean, cov);

        Eigen::VectorXd interiorPoint(cols);
        interiorPoint << 1.46, 3.54;

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        model)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        double num_samples = 10'000;
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            BOOST_CHECK(std::isfinite(state(0)));
            BOOST_CHECK(std::isfinite(state(1)));
        }
    };

    BOOST_AUTO_TEST_CASE(GaussianOnCrown) {
        auto A = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../../../resources/GaussianCrown/A.csv");
        auto b = hops::CsvReader::readVector<Eigen::VectorXd>("../../../../resources/GaussianCrown/b.csv");
        auto mean = hops::CsvReader::readVector<Eigen::VectorXd>("../../../../resources/GaussianCrown/mean.csv");
        auto cov = hops::CsvReader::readMatrix<Eigen::MatrixXd>("../../../../resources/GaussianCrown/cov.csv");
        auto interiorPoint = hops::CsvReader::readVector<Eigen::VectorXd>("../../../../resources/GaussianCrown/interiorPoint.csv");

        hops::Gaussian model(mean, cov);

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::TruncatedGaussianProposal(A,
                                                        b,
                                                        interiorPoint,
                                                        model)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        double num_samples = 25'000;
        for (int i = 0; i < num_samples; ++i) {
            auto[alpha, state] = mc.draw(randomNumberGenerator);
            BOOST_CHECK(std::isfinite(state(0)));
            BOOST_CHECK(std::isfinite(state(1)));
            BOOST_REQUIRE(((b - A * state).array() >= 0).all());
        }
    }

BOOST_AUTO_TEST_SUITE_END()
