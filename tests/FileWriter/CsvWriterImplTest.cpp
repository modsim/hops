#include <Eigen/Core>
#include <gtest/gtest.h>
#include <hops/FileWriter/CsvWriterImpl.hpp>

namespace {
    TEST(CsvWriterImpl, writeInts) {
        std::vector<int> data{-525, 0, 35};
        std::string expectedOutput = "-525\n0\n35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeLongs) {
        std::vector<long> data{-525, 0, 35};
        std::string expectedOutput = "-525\n0\n35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeFloats) {
        std::vector<float> data{1.25, 5.513, 8.35};
        std::string expectedOutput = "1.25\n5.513\n8.35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeDoubles) {
        std::vector<double> data{1.25, 5.513, 8.35};
        std::string expectedOutput = "1.25\n5.513\n8.35\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeStrings) {
        std::vector<std::string> data{"test1", "test2"};
        std::string expectedOutput = "test1\ntest2\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeOneDimensionalRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeEigenFloatVectors) {
        std::vector<Eigen::VectorXf> data;
        Eigen::VectorXf v1(3);
        v1 << 1.1,5.6,3.14;
        data.emplace_back(v1);
        Eigen::VectorXf v2(1);
        v2 << -1.1;
        data.emplace_back(v2);
        std::string expectedOutput = "1.1,5.6,3.14\n-1.1\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeEigenVectorRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }

    TEST(CsvWriterImpl, writeEigenDoubleVectors) {
        std::vector<Eigen::VectorXd> data;
        Eigen::VectorXd v1(3);
        v1 << 1.1,5.6,3.14;
        data.emplace_back(v1);
        Eigen::VectorXd v2(1);
        v2 << -1.1;
        data.emplace_back(v2);
        std::string expectedOutput = "1.1,5.6,3.14\n-1.1\n";

        std::ostringstream actualOutput;
        hops::internal::CsvWriterImpl::writeEigenVectorRecords(actualOutput, data);

        EXPECT_EQ(actualOutput.str(), expectedOutput);
    }
}