#define BOOST_TEST_MODULE ChainDataTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <iostream>
#include <memory>

#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(ChainDataTestSuite)
    BOOST_AUTO_TEST_CASE(InitializeAndRead) {
        Eigen::MatrixXd A(4, 2);
        Eigen::VectorXd b(4);
        Eigen::VectorXd x(2);

        A <<  1,  0,
              0,  1,
             -1,  0,
              0, -1;
        b << 1, 1, 0, 0;
        x << 0.5, 0.5;

        hops::ChainData chainData;
        long numberOfSamples = 1000;

        if (true) {
            hops::RandomNumberGenerator randomNumberGenerator;
            std::unique_ptr<hops::MarkovChain> mc = hops::MarkovChainFactory::createMarkovChain(hops::MarkovChainType::HitAndRun, A, b, x);
            mc->installDataObject(chainData);

            mc->draw(randomNumberGenerator, numberOfSamples);
        }

        BOOST_CHECK_EQUAL(chainData.getAcceptanceRates().size(), numberOfSamples);
        BOOST_CHECK_EQUAL(chainData.getStates().size(), numberOfSamples);
        BOOST_CHECK_EQUAL(chainData.getTimestamps().size(), numberOfSamples);
    }
BOOST_AUTO_TEST_SUITE_END()
