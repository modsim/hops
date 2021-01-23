#ifndef HOPS_HDF5READER_HPP
#define HOPS_HDF5READER_HPP

#include <fstream>
#include <string>
#include <vector>

namespace hops {
    class Hdf5Reader {
    public:
        Hdf5Reader() = delete;

        /**
         * @brief Reads an object of Type T from a dataset in an hdf5 file.
         * @tparam T
         * @param file hdf5 file
         * @param pathToDataset path to dataset in the hdf5 file. Usually starts with /, but it is not required.
         * @return
         */
        template<typename T>
        static T read(const std::string &file, const std::string &pathToDataset);
    };
}

#endif //HOPS_HDF5READER_HPP
