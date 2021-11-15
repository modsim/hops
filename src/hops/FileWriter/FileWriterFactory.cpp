#include "FileWriterFactory.hpp"
#include "FileWriterType.hpp"
#include "CsvWriter.hpp"

std::unique_ptr<hops::FileWriter>
hops::FileWriterFactory::createFileWriter(const std::string &filename, FileWriterType fileWriterType) {
    switch(fileWriterType) {
        case FileWriterType::CSV: {
            return std::make_unique<CsvWriter>(filename);
        }
        case FileWriterType::HDF5: {
            return nullptr;
        }
        default:
            throw std::runtime_error("Invalid Parameter for FileWriterType.");
    }
}
