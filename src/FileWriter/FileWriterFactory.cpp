#include <hops/FileWriter/FileWriterFactory.hpp>
#include <hops/FileWriter/FileWriterType.hpp>
#include <hops/FileWriter/CsvWriter.hpp>
#include <hops/FileWriter/Hdf5Writer.hpp>

std::unique_ptr<const hops::FileWriter>
hops::FileWriterFactory::createFileWriter(const std::string &filename, FileWriterType fileWriterType) {
    switch(fileWriterType) {
        case FileWriterType::CSV: {
            return std::make_unique<CsvWriter>(filename);
        }
        case FileWriterType::HDF5: {
            return std::make_unique<Hdf5Writer>(filename);
        }
        default:
            throw std::runtime_error("Invalid Parameter for FileWriterType.");
    }
}
