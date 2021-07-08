#define BOOST_TEST_MODULE DataTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <hops/hops.hpp>


BOOST_AUTO_TEST_SUITE(DataTestSuite)

    BOOST_AUTO_TEST_CASE(ChainDataTracking) {
        Eigen::MatrixXd A(4, 2);
        Eigen::VectorXd b(4);
        Eigen::VectorXd x(2);

        A << 1, 0,
                0, 1,
                -1, 0,
                0, -1;
        b << 1, 1, 0, 0;
        x << 0.5, 0.5;

        hops::ChainData chainData;
        long numberOfSamples = 1000;

        if (true) {
            hops::RandomNumberGenerator randomNumberGenerator;
            std::unique_ptr<hops::MarkovChain> mc = hops::MarkovChainFactory::createMarkovChain(
                    hops::MarkovChainType::HitAndRun, A, b, x);
            mc->installDataObject(chainData);

            mc->draw(randomNumberGenerator, numberOfSamples);
        }

        BOOST_CHECK_EQUAL(chainData.getAcceptanceRates().size(), numberOfSamples);
        BOOST_CHECK_EQUAL(chainData.getStates().size(), numberOfSamples);
        BOOST_CHECK_EQUAL(chainData.getTimestamps().size(), numberOfSamples);
    }

    BOOST_AUTO_TEST_CASE(SetProblemAndRun) {
        unsigned dimension = 2;
        Eigen::MatrixXd A(4, dimension);
        Eigen::VectorXd b(4);
        Eigen::VectorXd x(dimension);

        A << 1, 0,
                0, 1,
                -1, 0,
                0, -1;
        b << 1, 1, 0, 0;
        x << 0.5, 0.5;

        unsigned long numberOfSamples = 1000;

        hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>
                model(Eigen::VectorXd::Zero(dimension), Eigen::MatrixXd::Identity(dimension, dimension));

        hops::Problem problem(A, b, model);
        problem.setStartingPoint(x);
        hops::Run<decltype(model)> run(problem, hops::MarkovChainType::HitAndRun, numberOfSamples);

        run.sample();

        BOOST_CHECK_EQUAL(run.getData().getStates()[0]->size(), numberOfSamples);
    }

    BOOST_AUTO_TEST_CASE(CopyRun) {
        unsigned dimension = 2;
        Eigen::MatrixXd A(4, dimension);
        Eigen::VectorXd b(4);
        Eigen::VectorXd x(dimension);

        A << 1, 0,
                0, 1,
                -1, 0,
                0, -1;
        b << 1, 1, 0, 0;
        x << 0.5, 0.5;

        hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>
                model(Eigen::VectorXd::Zero(dimension), Eigen::MatrixXd::Identity(dimension, dimension));

        BOOST_CHECK_NO_THROW(
                hops::Problem problem(A, b, model);
                problem.setStartingPoint(x);
                hops::Run<decltype(model)> run(problem);

                hops::Run<decltype(model)> run2(run);
        );
    }

    BOOST_AUTO_TEST_CASE(ComputeStatisticsAfterRun) {
        unsigned dimension = 2;
        Eigen::MatrixXd A(4, dimension);
        Eigen::VectorXd b(4);
        Eigen::VectorXd x(dimension);

        A << 1, 0,
                0, 1,
                -1, 0,
                0, -1;
        b << 1, 1, 0, 0;
        x << 0.5, 0.5;

        unsigned long numberOfSamples = 10;

        hops::MultivariateGaussianModel<Eigen::MatrixXd, Eigen::VectorXd>
                model(Eigen::VectorXd::Zero(dimension), Eigen::MatrixXd::Identity(dimension, dimension));

        hops::Problem problem(A, b, model);
        problem.setStartingPoint(x);
        hops::Run<decltype(model)> run(problem, hops::MarkovChainType::HitAndRun, numberOfSamples);

        run.sample();

        BOOST_CHECK_EQUAL(run.getData().getStates()[0]->size(), numberOfSamples);

        run.getData().computeExpectedSquaredJumpDistance();

        double expectedResult = 3.20334;
        double actualResult = hops::computeExpectedSquaredJumpDistance(run.getData())(0);

        BOOST_CHECK_CLOSE(expectedResult, actualResult, 0.01);
    }
}
