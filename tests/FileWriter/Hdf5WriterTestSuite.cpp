#define BOOST_TEST_MODULE Hdf5WriterTestSuite
#define BOOST_TEST_DYN_LINK

#include <boost/test/included/unit_test.hpp>
#include <Eigen/Core>
#include <hops/hops.hpp>

BOOST_AUTO_TEST_SUITE(Hdf5Writer)

    BOOST_AUTO_TEST_CASE( writeVectorOfFloats) {
        std::vector<float> data = {1, 2, 3, -.3, 5.4, 3.13, 1510};
        hops::Hdf5Writer writer("writeVectorOfFloats");
        writer.write("floatData", data);
    }

    BOOST_AUTO_TEST_CASE( writeVectorOfDoubles) {
        std::vector<double> data = {1, 2, 3, -.3, 5.4, 3.13, 1510};
        hops::Hdf5Writer writer("writeVectorOfDoubles");
        writer.write("doubleData", data);
    }

    BOOST_AUTO_TEST_CASE( writeVectorOfLongs) {
        std::vector<long> data = {1, 2, 3, 1510};
        hops::Hdf5Writer writer("writeVectorOfLongs");
        writer.write("longData", data);
    }

    BOOST_AUTO_TEST_CASE( writeVectorOfEigenVectorXf) {
        std::vector<Eigen::VectorXf> data;
        for (int i = 0; i < 10; ++i) {
            Eigen::VectorXf x = (1 + i) * Eigen::VectorXf::Ones(5);
            data.emplace_back(x);
        }
        hops::Hdf5Writer writer("writeVectorOfEigenVectorXf");
        writer.write("EigenVectors", data);
    }

    BOOST_AUTO_TEST_CASE( writeVectorOfEigenVectorXd) {
        std::vector<Eigen::VectorXd> data;
        for (int i = 0; i < 10; ++i) {
            Eigen::VectorXd x = (1 + i) * Eigen::VectorXd::Ones(5);
            data.emplace_back(x);
        }
        hops::Hdf5Writer writer("writeVectorOfEigenVectorXd");
        writer.write("EigenVectors", data);
    }

    BOOST_AUTO_TEST_CASE( writeVectorOfStrings) {
        std::vector<std::string> data = {"string1", "string2", "string3", "string4", "string5", "string6"};
        hops::Hdf5Writer writer("writeVectorOfStrings");
        writer.write("strings", data);
    }

    BOOST_AUTO_TEST_CASE( writeEigenMatrixXd) {
        Eigen::MatrixXd data = 5.5 * Eigen::MatrixXd::Ones(3, 5);
        hops::Hdf5Writer writer("writeEigenMatrixXd");
        writer.write("EigenMatrix", data);
    }

    BOOST_AUTO_TEST_CASE( writeEigenVectorXd) {
        Eigen::VectorXd data = 5.5 * Eigen::VectorXd::Ones(5);
        hops::Hdf5Writer writer("writeEigenVectorXd");
        writer.write("EigenVector", data);
    }

BOOST_AUTO_TEST_SUITE_END()
