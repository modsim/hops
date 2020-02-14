#ifndef HOPS_CSVREADER_HPP
#define HOPS_CSVREADER_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <fstream>
#include <string>
#include <vector>

namespace hops {
    class CsvReader {
    public:
        template<typename VectorType>
        static VectorType readVector(const std::string &file);

        template<typename MatrixType>
        static MatrixType readMatrix(const std::string &file);
    };
}

#endif //HOPS_CSVREADER_HPP
