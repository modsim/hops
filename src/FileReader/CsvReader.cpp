#include "nups/FileReader/CsvReader.hpp"

template<typename VectorType>
VectorType nups::CsvReader::readVector(const std::string &file) {
    std::ifstream fileStream(file);
    if (fileStream.fail()) {
        throw std::runtime_error("Could not access file: " + file);
    }
    std::vector<std::string> cells;
    std::string line;
    while (std::getline(fileStream, line)) {
        cells.push_back(line);
    }

    VectorType result(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        result(i) = std::stod(cells[i]);
    }
    return result;
}

template Eigen::VectorXi nups::CsvReader::readVector(const std::string &file);

template Eigen::Matrix<long, Eigen::Dynamic, 1> nups::CsvReader::readVector(const std::string &file);

template Eigen::VectorXf nups::CsvReader::readVector(const std::string &file);

template Eigen::VectorXd nups::CsvReader::readVector(const std::string &file);


template<typename MatrixType>
MatrixType nups::CsvReader::readMatrix(const std::string &file) {

    std::ifstream fileStream(file);
    if (fileStream.fail()) {
        throw std::runtime_error("Could not access file: " + file);
    }
    std::string line;
    std::vector<std::vector<std::string>> stringRepresentationOfMatrix;
    while (std::getline(fileStream, line)) {
        std::stringstream lineStream(line);
        std::vector<std::string> cells;
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            cells.push_back(cell);
        }
        stringRepresentationOfMatrix.push_back(cells);
    }

    Eigen::SparseMatrix<typename MatrixType::Scalar> result(stringRepresentationOfMatrix.size(),
                                                            stringRepresentationOfMatrix.at(0).size());
    std::vector<Eigen::Triplet<typename MatrixType::Scalar>> triplets;

    for (size_t i = 0; i < stringRepresentationOfMatrix.size(); ++i) {
        for (size_t j = 0; j < stringRepresentationOfMatrix.at(i).size(); ++j) {
            typename MatrixType::Scalar value = std::stod(stringRepresentationOfMatrix.at(i).at(j));
            if (value != 0) {
                triplets.emplace_back(i, j, value);
            }
        }
    }
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

template Eigen::MatrixXi nups::CsvReader::readMatrix(const std::string &file);

template Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic> nups::CsvReader::readMatrix(const std::string &file);

template Eigen::MatrixXf nups::CsvReader::readMatrix(const std::string &file);

template Eigen::MatrixXd nups::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<int> nups::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<long> nups::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<float> nups::CsvReader::readMatrix(const std::string &file);

template Eigen::SparseMatrix<double> nups::CsvReader::readMatrix(const std::string &file);
