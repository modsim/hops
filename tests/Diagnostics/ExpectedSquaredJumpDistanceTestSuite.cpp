#define BOOST_TEST_MODULE ExpectecSquaredJumpDistanceTestSuite

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(ExpectedSquaredJumpDistanceTestSuite)
    BOOST_AUTO_TEST_CASE(ComputeAllDraws) {
        double expectedResult = 2./3.;
        std::vector<double> chain1{0, 1, 1, 2};
        std::vector<double> chain2{0, 0, 1, 2};

        std::vector<std::vector<Eigen::VectorXd>> chains(2);

        for (size_t i = 0; i < chain1.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain1[i];
            chains[0].push_back(draw);
        }

        for (size_t i = 0; i < chain2.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain2[i];
            chains[1].push_back(draw);
        }

        for (size_t i = 0; i < chains.size(); ++i) {
            double esjd = hops::computeExpectedSquaredJumpDistance(chains[i]);
            BOOST_CHECK_CLOSE(expectedResult, esjd, 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeAllDrawsPointerArray) {
        double expectedResult = 2./3.;
        std::vector<Eigen::VectorXd> chain1;
        std::vector<Eigen::VectorXd> chain2;
        
        for (auto& i : {0, 1, 1, 2}) {
            Eigen::VectorXd x(1);
            x(0) = i;
            chain1.push_back(x);
        }

        for (auto& i : {0, 0, 1, 2}) {
            Eigen::VectorXd x(1);
            x(0) = i;
            chain2.push_back(x);
        }

        std::vector<const std::vector<Eigen::VectorXd>*> chains;

        chains.push_back(&chain1);
        chains.push_back(&chain2);

        auto esjds = hops::computeExpectedSquaredJumpDistance(chains);

        for (size_t i = 0; i < chains.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult, esjds[i], 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeIncrementally) {
        std::vector<double> expectedResult{1, 0};
        std::vector<double> chain1{0, 1};
        std::vector<double> chain2{0, 0};

        std::vector<std::vector<Eigen::VectorXd>> chains(2);

        for (size_t i = 0; i < chain1.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain1[i];
            chains[0].push_back(draw);
        }

        for (size_t i = 0; i < chain2.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain2[i];
            chains[1].push_back(draw);
        }

        std::vector<double> esjds;
        for (size_t i = 0; i < chains.size(); ++i) {
            double esjd = hops::computeExpectedSquaredJumpDistance(chains[i]);
            esjds.push_back(esjd);
            BOOST_CHECK_CLOSE(expectedResult[i], esjd, 0.01);
        }
        
        chain1 = std::vector<double>({1, 2});
        chain2 = std::vector<double>({1, 2});

        for (size_t i = 0; i < chain1.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain1[i];
            chains[0].push_back(draw);
        }

        for (size_t i = 0; i < chain2.size(); ++i) {
            Eigen::VectorXd draw(1);
            draw(0) = chain2[i];
            chains[1].push_back(draw);
        }

        for (size_t i = 0; i < chains.size(); ++i) {
            double esjd = hops::computeExpectedSquaredJumpDistance(chains[i], 2, esjds[i], 2);
            BOOST_CHECK_CLOSE(2./3., esjd, 0.01);
        }
    }

    BOOST_AUTO_TEST_CASE(ComputeIncrementallyPointerArray) {
        std::vector<double> expectedResult{1, 0};
        std::vector<const std::vector<Eigen::VectorXd>*> chains;

        std::vector<Eigen::VectorXd>chain1;
        std::vector<Eigen::VectorXd>chain2;

        for (auto& i : {0, 1}) {
            Eigen::VectorXd draw(1);
            draw(0) = i;
            chain1.push_back(draw);
        }

        for (auto& i : {0, 0}) {
            Eigen::VectorXd draw(1);
            draw(0) = i;
            chain2.push_back(draw);
        }

        chains.push_back(&chain1);
        chains.push_back(&chain2);

        auto esjds = hops::computeExpectedSquaredJumpDistance(chains);

        for (size_t i = 0; i < chains.size(); ++i) {
            BOOST_CHECK_CLOSE(expectedResult[i], esjds[i], 0.01);
        }
        
        for (auto& i : {1, 2}) {
            Eigen::VectorXd draw(1);
            draw(0) = i;
            chain1.push_back(draw);
            chain2.push_back(draw);
        }

        esjds = hops::computeExpectedSquaredJumpDistance(chains, 2, esjds, 2);
        for (size_t i = 0; i < chains.size(); ++i) {
            BOOST_CHECK_CLOSE(2./3., esjds[i], 0.01);
        }
    }
BOOST_AUTO_TEST_SUITE_END()
