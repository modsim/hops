#define BOOST_TEST_MODULE ProposalStatisticsTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <hops/MarkovChain/Proposal/ProposalStatistics.hpp>

BOOST_AUTO_TEST_SUITE(ProposalStatistics)

    BOOST_AUTO_TEST_CASE(TestSimpleStatistics) {
        std::vector<std::string> names {"x1", "x2", "x1", "x3"};
        std::vector<double> values {1.2, 3, 4, 5};

        std::unordered_map<std::string, std::vector<double>> expectedStatistics =
                {
                        {"x1", {1.2, 4}},
                        {"x2", {3}},
                        {"x3", {5}},

                };

        hops::ProposalStatistics infos;
        for(size_t i=0; i<names.size(); i++) {
            infos.appendInfo(names[i], values[i]);
        }
        auto actualStatistics = infos.getStatistics();

        BOOST_CHECK(actualStatistics == expectedStatistics);
    }

BOOST_AUTO_TEST_SUITE_END()

