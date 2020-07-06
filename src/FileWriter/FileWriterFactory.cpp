#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/FileWriter/CsvWriter.hpp>

std::unique_ptr<const hops::FileWriter>
hops::FileWriterFactory::createFileWriter(const std::string &filename, FileWriterType fileWriterType) {
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
