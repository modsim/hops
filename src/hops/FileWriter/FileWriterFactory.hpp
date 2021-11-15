#ifndef HOPS_FILEWRITERFACTORY_HPP
#define HOPS_FILEWRITERFACTORY_HPP

#include "FileWriter.hpp"
#include "FileWriterType.hpp"
#include <memory>
#include <string>

namespace hops {
    class FileWriterFactory {
    public:
        static std::unique_ptr<FileWriter>
        createFileWriter(const std::string &filename, FileWriterType fileWriterType);
    };
}

#endif //HOPS_FILEWRITERFACTORY_HPP
