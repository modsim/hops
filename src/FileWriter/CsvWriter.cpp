#include <experimental/filesystem>
#include <fstream>
#include <nups/FileWriter/CsvWriter.hpp>
#include "nups/FileWriter/CsvWriterImpl.hpp"

// TODO fix hard-coded precision

nups::CsvWriter::CsvWriter(std::string path) : path(std::move(path)) {
    std::experimental::filesystem::create_directories(CsvWriter::path);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<float> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<double> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<long> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeEigenVectorRecords(out, records);
}

void nups::CsvWriter::write(const std::string &description, const std::vector<std::string> &records) const {
    std::experimental::filesystem::path outputPath(CsvWriter::path);
    outputPath /= CsvWriter::path + "_" + description + ".csv";
    std::ofstream out(outputPath.string(), std::ios_base::app);
    out.precision(17);
    internal::CsvWriterImpl::writeOneDimensionalRecords(out, records);
}
