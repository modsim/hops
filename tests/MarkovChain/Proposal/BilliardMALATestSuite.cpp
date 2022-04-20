#define BOOST_TEST_MODULE BilliardMALAProposalTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <Eigen/Core>

#include <hops/MarkovChain/MarkovChainAdapter.hpp>
#include <hops/MarkovChain/Draw/MetropolisHastingsFilter.hpp>
#include <hops/MarkovChain/Proposal/BilliardMALAProposal.hpp>
#include <hops/Model/Rosenbrock.hpp>
#include <hops/Model/Gaussian.hpp>
#include <hops/RandomNumberGenerator/RandomNumberGenerator.hpp>
#include <hops/Statistics/EffectiveSampleSize.hpp>

BOOST_AUTO_TEST_SUITE(BilliardMALAProposal)

    BOOST_AUTO_TEST_CASE(RosenBrockInCube) {
        const long rows = 8;
        const long cols = 4;
        Eigen::MatrixXd A(rows, cols);
        A << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1,
                -1, 0, 0, 0,
                0, -1, 0, 0,
                0, 0, -1, 0,
                0, 0, 0, -1;
        Eigen::VectorXd b(rows);
        b << 1, 1, 1, 1, 1, 1, 1, 1;
        Eigen::VectorXd interiorPoint(cols);
        for (size_t i = 0; i < cols; ++i) {
            interiorPoint(i) = 0;
        }

        auto model = hops::Rosenbrock(1, Eigen::VectorXd::Zero(cols / 2));

        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);


        hops::RandomNumberGenerator randomNumberGenerator(42);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;

        BOOST_CHECK(proposer.getModel() != nullptr);
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


        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);

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

    BOOST_AUTO_TEST_CASE(ColdnessGaussianInCube) {
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

        auto model = hops::Coldness(hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols)), 0.5);


        long max_reflections = 10;
        hops::BilliardMALAProposal proposer(A,
                                            b,
                                            interiorPoint,
                                            model,
                                            max_reflections);

        hops::RandomNumberGenerator randomNumberGenerator(42);
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
            double acceptanceChance = proposer.computeLogAcceptanceProbability();
            BOOST_CHECK(((b - A * proposal).array() > 0).all());
            BOOST_CHECK(std::exp(acceptanceChance) >= 0);
            proposer.acceptProposal();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;

        Eigen::VectorXd proposal = proposer.propose(randomNumberGenerator);
        Eigen::VectorXd expectedProposal(3);
        expectedProposal << -0.381175410126282255, 0.99015677158076798, -0.200087788734234273;
        BOOST_CHECK(proposal.isApprox(expectedProposal));

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

        auto model = hops::Gaussian(interiorPoint, Eigen::MatrixXd::Identity(cols, cols));

        auto mc = hops::MarkovChainAdapter(
                hops::MetropolisHastingsFilter(
                        hops::BilliardMALAProposal(A,
                                                   b,
                                                   interiorPoint,
                                                   model,
                                                   100)
                )
        );

        hops::RandomNumberGenerator randomNumberGenerator(42);
        std::vector<Eigen::VectorXd> samples;
        double num_samples = 25'000;
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

        BOOST_CHECK(std::abs(mean - model.getMean()(0)) < 2 * standardErrorOfMean);
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
                        hops::BilliardMALAProposal(A,
                                                   b,
                                                   interiorPoint,
                                                   gaussian,
                                                   100)
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



BOOST_AUTO_TEST_SUITE_END()
