#ifndef HOPS_SBMLREADER_HPP
#define HOPS_SBMLREADER_HPP

#include <hops/FileReader/SbmlModel.hpp>
#include <string>
#include <vector>

namespace hops {
    class SbmlReader {
    public:
        SbmlReader() = delete;

        /**
         * @brief parses linear model constraints and stores them to  an SbmlModel object.
         * @tparam MatrixType
         * @tparam VectorType
         * @param file
         * @return
         */
        template<typename MatrixType, typename VectorType>
        static SbmlModel<MatrixType, VectorType> readModel(const std::string &file);
    };
}

#endif //HOPS_SBMLREADER_HPP
