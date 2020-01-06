#include <nups/FileWriter/FileWriterFactory.hpp>
#include <nups/FileWriter/FileWriterType.hpp>
#include <nups/FileWriter/CsvWriter.hpp>

std::unique_ptr<const nups::FileWriter>
nups::FileWriterFactory::createFileWriter(const std::string &filename, FileWriterType fileWriterType) {
    switch(fileWriterType) {
        case FileWriterType::Csv: {
            return std::make_unique<CsvWriter>(filename);
        }
        case FileWriterType::Hdf5: {
            // TODO
            std::unique_ptr<const FileWriter> f;
            return f;
        }
        default:
            throw std::runtime_error("Invalid Parameter for FileWriterType.");
    }
}
