#define BOOST_TEST_MODULE SbmlReaderTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(SbmlReader)

    BOOST_AUTO_TEST_CASE(read_e_coli_core) {
        auto expectedStoichiometry = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/e_coli_core/S_e_coli_core.csv");
        auto expectedUb = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/e_coli_core/ub_e_coli_core.csv");
        auto expectedLb = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/e_coli_core/lb_e_coli_core.csv");
        auto e_coli_core = hops::SbmlReader::readModel<Eigen::MatrixXd, Eigen::VectorXd>(
                "../../resources/e_coli_core/e_coli_core.xml");

        Eigen::MatrixXd actualStoichiometry = e_coli_core.getStoichiometry();
        Eigen::VectorXd actualLb = e_coli_core.getLowerBounds();
        Eigen::VectorXd actualUb = e_coli_core.getUpperBounds();

        BOOST_CHECK(expectedStoichiometry == actualStoichiometry);
        BOOST_CHECK(expectedUb == actualUb);
        BOOST_CHECK(expectedLb == actualLb);
    }

BOOST_AUTO_TEST_SUITE_END()
