#include <experimental/filesystem>
#include <fstream>
#include <hops/FileWriter/CsvWriter.hpp>
#include "hops/FileWriter/CsvWriterImpl.hpp"

// TODO replace hard-coded precision by user choice

hops::CsvWriter::CsvWriter(std::string path) : path(std::move(path)) {
    std::experimental::filesystem::create_directories(CsvWriter::path);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<float> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<double> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<long> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const std::vector<std::string> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void hops::CsvWriter::write(const std::string &description, const Eigen::MatrixXd &matrix) {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    // TODO move implementaiton to writerImpl
    for (long i = 0; i < matrix.rows(); ++i) {
        for (long j = 0; j < matrix.cols(); ++j) {
            out << matrix(i, j);
            if (j != matrix.cols() - 1) {
                out << ",";
            }
        }
        out << "\n";
    }
}

void hops::CsvWriter::write(const std::string &description, const Eigen::VectorXd &vector) {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= outputPath.filename().string() + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    // TODO move implementaiton to writerImpl
    for (long i = 0; i < vector.rows(); ++i) {
        out << vector(i);
        out << "\n";
    }
}
