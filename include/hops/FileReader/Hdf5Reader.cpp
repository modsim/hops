#include "hops/FileReader/Hdf5Reader.hpp"
#include <highfive/H5Easy.hpp>
#include <Eigen/Core>
#include <Eigen/Sparse>

template<typename T>
T hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset) {
    H5Easy::File hdfFile(file, H5Easy::File::ReadOnly);
    if(hdfFile.exist(pathToDataset)) {
        auto result = H5Easy::load<T>(hdfFile, pathToDataset);
        return result;
    }
    throw std::runtime_error("DataSet does not exist.");
}

template std::string hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template int hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template float hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template double hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template std::vector<std::string> hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::VectorXi hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::VectorXf hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::VectorXd hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::MatrixXi hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::MatrixXf hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template Eigen::MatrixXd hops::Hdf5Reader::read(const std::string &file, const std::string &pathToDataset);

template<>
Eigen::SparseMatrix<float>
hops::Hdf5Reader::read<Eigen::SparseMatrix<float>>(const std::string &file, const std::string &pathToDataset) {
    return hops::Hdf5Reader::read<Eigen::MatrixXf>(file, pathToDataset).sparseView();
}

template<>
Eigen::SparseMatrix<double>
hops::Hdf5Reader::read<Eigen::SparseMatrix<double>>(const std::string &file, const std::string &pathToDataset) {
    return hops::Hdf5Reader::read<Eigen::MatrixXd>(file, pathToDataset).sparseView();
}

template<>
Eigen::SparseMatrix<int>
hops::Hdf5Reader::read<Eigen::SparseMatrix<int>>(const std::string &file, const std::string &pathToDataset) {
    return hops::Hdf5Reader::read<Eigen::MatrixXi>(file, pathToDataset).sparseView();
}

