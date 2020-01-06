#ifndef NUPS_FILEWRITERFACTORY_HPP
#define NUPS_FILEWRITERFACTORY_HPP

#include <nups/FileWriter/FileWriter.hpp>
#include <nups/FileWriter/FileWriterType.hpp>
#include <memory>
#include <string>

namespace nups {
    class FileWriterFactory {
    public:
        static std::unique_ptr<const FileWriter>
        createFileWriter(const std::string &filename, FileWriterType fileWriterType);
    };
}

#endif //NUPS_FILEWRITERFACTORY_HPP
