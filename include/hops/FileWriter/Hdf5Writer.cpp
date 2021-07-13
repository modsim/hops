#include <highfive/H5Easy.hpp>
#include <highfive/H5File.hpp>
#include <iostream>

#include "hops/FileWriter/Hdf5Writer.hpp"

hops::Hdf5Writer::Hdf5Writer(std::string path) : path(std::move(path)) {
    Hdf5Writer::path += ".hdf5";
}

hops::Hdf5Writer::~Hdf5Writer() = default;

void hops::Hdf5Writer::write(const std::string &description, const std::vector<float> &records) const {
    H5Easy::File file(Hdf5Writer::path,
    H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::DataSet dataset = file.exist(datasetName) ? file.getDataSet(datasetName) :
                              file.createDataSet<float>(datasetName, H5Easy::DataSpace::From(records));
    dataset.write(records);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<double> &records) const {
    H5Easy::File file(Hdf5Writer::path,
    H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::DataSet dataset = file.exist(datasetName) ? file.getDataSet(datasetName) :
                              file.createDataSet<double>(datasetName, H5Easy::DataSpace::From(records));
    dataset.write(records);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<long> &records) const {
    H5Easy::File file(Hdf5Writer::path,
    H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::DataSet dataset = file.exist(datasetName) ? file.getDataSet(datasetName) :
                              file.createDataSet<long>(datasetName, H5Easy::DataSpace::From(records));
    dataset.write(records);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector <Eigen::VectorXf> &records) const {
    if(records.empty()) {
        return;
    }
    H5Easy::File
            file(Hdf5Writer::path, H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    Eigen::MatrixXf output(records.size(), records[0].rows());
    for(size_t i=0; i<records.size(); ++i) {
        output.row(i) = records[i];
    }

    H5Easy::dump(file, datasetName, output, H5Easy::DumpMode::Create);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector <Eigen::VectorXd> &records) const {
    if(records.empty()) {
        return;
    }
    H5Easy::File
    file(Hdf5Writer::path, H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    Eigen::MatrixXd output(records.size(), records[0].rows());
    for(size_t i=0; i<records.size(); ++i) {
        output.row(i) = records[i];
    }

    H5Easy::dump(file, datasetName, output, H5Easy::DumpMode::Create);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector <std::string> &records) const {
    H5Easy::File
    file(Hdf5Writer::path, H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::dump(file, datasetName, records, H5Easy::DumpMode::Create);
}

void hops::Hdf5Writer::write(const std::string &description, const Eigen::MatrixXd &matrix) const {
    H5Easy::File
    file(Hdf5Writer::path, H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::dump(file, datasetName, matrix, H5Easy::DumpMode::Create);
}

void hops::Hdf5Writer::write(const std::string &description, const Eigen::VectorXd &vector) const {
    H5Easy::File
    file(Hdf5Writer::path, H5Easy::File::ReadWrite | H5Easy::File::Create | H5Easy::File::Truncate);
    std::string datasetName = "/" + description;
    H5Easy::dump(file, datasetName, vector, H5Easy::DumpMode::Create);
}
