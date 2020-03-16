#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/FileReader/CsvReader.hpp>
#include <hops/FileReader/SbmlModel.hpp>
#include <hops/FileReader/SbmlReader.hpp>

namespace {
    TEST(SbmlReader, read_e_coli_core) {
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

        EXPECT_EQ(expectedStoichiometry, actualStoichiometry);
        EXPECT_EQ(expectedUb, actualUb);
        EXPECT_EQ(expectedLb, actualLb);
    }

    TEST(SbmlReader, read_iJO1366) {
        auto expectedStoichiometry = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/iJO1366/S_iJO1366.csv");
        auto expectedUb = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/iJO1366/ub_iJO1366.csv");

        auto expectedLb = hops::CsvReader::readMatrix<Eigen::MatrixXd>(
                "../../resources/iJO1366/lb_iJO1366.csv");
        auto iJO1366 = hops::SbmlReader::readModel<Eigen::MatrixXd, Eigen::VectorXd>(
                "../../resources/iJO1366/iJO1366.xml");

        Eigen::MatrixXd actualStoichiometry = iJO1366.getStoichiometry();
        Eigen::VectorXd actualLb = iJO1366.getLowerBounds();
        Eigen::VectorXd actualUb = iJO1366.getUpperBounds();

        EXPECT_EQ(expectedStoichiometry, actualStoichiometry);
        EXPECT_EQ(expectedUb, actualUb);
        EXPECT_EQ(expectedLb, actualLb);
    }
}