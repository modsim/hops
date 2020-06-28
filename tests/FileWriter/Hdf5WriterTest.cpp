#include <gtest/gtest.h>
#include <Eigen/Core>
#include <hops/FileWriter/Hdf5Writer.hpp>

namespace {
    TEST(Hdf5Writer, writeVectorOfFloats) {
        std::vector<float> data = {1, 2, 3, -.3, 5.4, 3.13, 1510};
        hops::Hdf5Writer writer("writeVectorOfFloats");
        // Explicitly test double writes
        writer.write("floatData", data);
        writer.write("floatData", data);
    }

    TEST(Hdf5Writer, writeVectorOfDoubles) {
        std::vector<double> data = {1, 2, 3, -.3, 5.4, 3.13, 1510};
        hops::Hdf5Writer writer("writeVectorOfDoubles");
        writer.write("doubleData", data);
        writer.write("doubleData", data);
    }

    TEST(Hdf5Writer, writeVectorOfLongs) {
        std::vector<long> data = {1, 2, 3, 1510};
        hops::Hdf5Writer writer("writeVectorOfLongs");
        writer.write("longData", data);
        writer.write("longData", data);
    }

    TEST(Hdf5Writer, writeVectorOfLongDoubles) {
        std::vector<long double> data = {1, 2, 3, -.3, 5.4, 3.13, 1510};
        hops::Hdf5Writer writer("writeVectorOfLongDoubles");
        writer.write("longDoubleData", data);
        writer.write("longDoubleData", data);
    }

    TEST(Hdf5Writer, writeVectorOfEigenVectorXf) {
        std::vector<Eigen::VectorXf> data;
        for (int i = 0; i < 10; ++i) {
            Eigen::VectorXf x = (1 + i) * Eigen::VectorXf::Ones(5);
            data.emplace_back(x);
        }
        hops::Hdf5Writer writer("writeVectorOfEigenVectorXf");
        writer.write("EigenVectors", data);
        writer.write("EigenVectors", data);
    }

    TEST(Hdf5Writer, writeVectorOfEigenVectorXd) {
        std::vector<Eigen::VectorXd> data;
        for (int i = 0; i < 10; ++i) {
            Eigen::VectorXd x = (1 + i) * Eigen::VectorXd::Ones(5);
            data.emplace_back(x);
        }
        hops::Hdf5Writer writer("writeVectorOfEigenVectorXd");
        writer.write("EigenVectors", data);
        writer.write("EigenVectors", data);
    }

    TEST(Hdf5Writer, writeVectorOfStrings) {
        std::vector<std::string> data = {"string1", "string2", "string3", "string4", "string5", "string6"};
        hops::Hdf5Writer writer("writeVectorOfStrings");
        writer.write("strings", data);
        writer.write("strings", data);
    }

    TEST(Hdf5Writer, writeEigenMatrixXd) {
        Eigen::MatrixXd data = 5.5 * Eigen::MatrixXd::Ones(3, 5);
        hops::Hdf5Writer writer("writeEigenMatrixXd");
        writer.write("EigenMatrix", data);
        writer.write("EigenMatrix", data);
    }

    TEST(Hdf5Writer, writeEigenVectorXd) {
        Eigen::VectorXd data = 5.5 * Eigen::VectorXd::Ones(5);
        hops::Hdf5Writer writer("writeEigenVectorXd");
        writer.write("EigenVector", data);
        writer.write("EigenVector", data);
    }
}
