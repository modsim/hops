#ifndef HOPS_CSVREADER_HPP
#define HOPS_CSVREADER_HPP

#include <fstream>
#include <string>
#include <vector>

namespace hops {
    class CsvReader {
    public:
        CsvReader() = delete;

        // TODO add choice of delimiter

        template<typename VectorType>
        static VectorType readVector(const std::string &file);

        template<typename MatrixType>
        static MatrixType readMatrix(const std::string &file, bool hasColumnAndRowNames=false);

        // TODO return names
//        template<typename VectorType>
//        static VectorType readVectorWithNames(const std::string &file);

        // TODO return names
//        template<typename MatrixType>
//        static MatrixType readMatrixWithNames(const std::string &file);
    };
}

#endif //HOPS_CSVREADER_HPP
