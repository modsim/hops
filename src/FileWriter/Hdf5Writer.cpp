#include "hops/FileWriter/Hdf5Writer.hpp"
#include <hdf5.h>
#include <iostream>

namespace {
    // file_id has to belong to an open file hdf5 file
    void createExtendableDataSetIfNotExist(hid_t file_id, int rank, const char *title) {
        hsize_t dims[2] = {0, 0};
        hsize_t max_dims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
        hsize_t chunk_dims[2] = {2, 100};

        hid_t fileSpace = H5Screate_simple(rank, dims, max_dims);
        if (fileSpace >= 0) {
            hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
            if (plist >= 0) {
                H5Pset_layout(plist, H5D_CHUNKED);
                H5Pset_chunk(plist, rank, chunk_dims);
                hid_t dataSet = H5Dcreate(file_id,
                                          title,
                                          H5T_NATIVE_DOUBLE,
                                          fileSpace,
                                          H5P_DEFAULT,
                                          plist,
                                          H5P_DEFAULT);
                H5Dclose(dataSet);
                H5Pclose(plist);
            }
        }
        H5Sclose(fileSpace);
    }

    template<typename DataPointer>
    void appendData(hid_t file,
                    const char *dataSetName,
                    long numberOfRows,
                    long numberOfCols,
                    DataPointer dataPointer,
                    hid_t data_mem_type
    ) {
        hsize_t dataCount[2];
        dataCount[0] = numberOfRows;
        dataCount[1] = numberOfCols;

        hid_t memorySpace = H5Screate_simple(2, dataCount, nullptr);
        hid_t dataSet = H5Dopen1(file, dataSetName);
        hid_t dataSpace = H5Dget_space(dataSet);
        hsize_t dimensions[2];
        H5Sget_simple_extent_dims(dataSpace, &dimensions[0], nullptr);
        hsize_t writeStart[2];
        writeStart[0] = dimensions[0];
        writeStart[1] = 0;
        hsize_t writeEnd[2];
        writeEnd[0] = writeStart[0] + dataCount[0];
        writeEnd[1] = writeStart[1] + dataCount[1];
        H5Dset_extent(dataSet, writeEnd);

        hid_t fileSpace = H5Dget_space(dataSet);
        H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, writeStart, nullptr, dataCount, nullptr);

        H5Dwrite(dataSet, data_mem_type, memorySpace, fileSpace, H5P_DEFAULT, dataPointer);

        H5Sclose(dataSpace);
        H5Sclose(fileSpace);
        H5Sclose(memorySpace);
        H5Dclose(dataSet);
    }

    template<typename Derived>
    void writeData(hid_t fileId, const std::string &description, long numberOfDataRows, long numberOfDataCols,
                   Derived data, hid_t hdf5_data_type) {
        int rank = numberOfDataCols == 1 ? 1 : 2;
        createExtendableDataSetIfNotExist(fileId, rank, description.c_str());
        appendData(fileId, description.c_str(), numberOfDataRows, numberOfDataCols, data, hdf5_data_type);
    }
}

hops::Hdf5Writer::Hdf5Writer(std::string path) : path(std::move(path)) {
    Hdf5Writer::path += ".h5";
    /* Turn off error handling permanently */
    H5Eset_auto1(NULL, NULL);
    fileId = H5Fcreate(Hdf5Writer::path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (fileId < 0) {
        throw std::runtime_error("Error creating or opening HDF5File " + path + ".");
    }
}

hops::Hdf5Writer::~Hdf5Writer() {
    H5Fclose(fileId);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<float> &records) const {
    writeData(Hdf5Writer::fileId, description, records.size(), 1, records.data(), H5T_NATIVE_FLOAT);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<double> &records) const {
    writeData(Hdf5Writer::fileId, description, records.size(), 1, records.data(), H5T_NATIVE_DOUBLE);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<long> &records) const {
    writeData(Hdf5Writer::fileId, description, records.size(), 1, records.data(), H5T_NATIVE_LONG);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<long double> &records) const {
    writeData(Hdf5Writer::fileId, description, records.size(), 1, records.data(), H5T_NATIVE_LDOUBLE);
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<Eigen::VectorXf> &records) const {
    if (!records.empty()) {
        // Reorganize data
        std::vector<float> data;
        data.reserve(records.size() * records[0].rows());
        for (const auto &record : records) {
            for (long i = 0; i < record.rows(); ++i) {
                data.emplace_back(record(i));
            }
        }
        writeData(Hdf5Writer::fileId, description, records.size(), records[0].rows(), data.data(), H5T_NATIVE_FLOAT);
    }
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<Eigen::VectorXd> &records) const {
    if (!records.empty()) {
        // Reorganize data
        std::vector<double> data;
        data.reserve(records.size() * records[0].rows());
        for (const auto &record : records) {
            for (long i = 0; i < record.rows(); ++i) {
                data.emplace_back(record(i));
            }
        }
        writeData(Hdf5Writer::fileId, description, records.size(), records[0].rows(), data.data(), H5T_NATIVE_DOUBLE);
    }
}

void hops::Hdf5Writer::write(const std::string &description,
                             const std::vector<Eigen::Matrix<long double, Eigen::Dynamic, 1>> &records) const {
    if (!records.empty()) {
        // Reorganize data
        std::vector<long double> data;
        data.reserve(records.size() * records[0].rows());
        for (const auto &record : records) {
            for (long i = 0; i < record.rows(); ++i) {
                data.emplace_back(record(i));
            }
        }
        writeData(Hdf5Writer::fileId, description, records.size(), records[0].rows(), data.data(), H5T_NATIVE_LDOUBLE);
    }
}

void hops::Hdf5Writer::write(const std::string &description, const std::vector<std::string> &records) const {
    hid_t fileType = H5Tcopy(H5T_C_S1);
    H5Tset_size(fileType, H5T_VARIABLE);
    hid_t memoryType = H5Tcopy(H5T_C_S1);
    H5Tset_size(memoryType, H5T_VARIABLE);

    hsize_t dims[1] = {records.size()};
    hid_t dataSpace = H5Screate_simple(1, dims, nullptr);

    hid_t dataSet = H5Dcreate1(fileId, description.c_str(), fileType, dataSpace, H5P_DEFAULT);

    std::vector<const char *> modelParameterNamesAsCString;
    modelParameterNamesAsCString.reserve(records.size());
    for (const auto &record: records) {
        modelParameterNamesAsCString.push_back(record.c_str());
    }

    H5Dwrite(dataSet, memoryType, H5S_ALL, H5S_ALL, H5P_DEFAULT, modelParameterNamesAsCString.data());
    H5Dclose(dataSet);
    H5Sclose(dataSpace);
    H5Tclose(fileType);
    H5Tclose(memoryType);
}

void hops::Hdf5Writer::write(const std::string &description, const Eigen::MatrixXd &matrix) const {
    writeData(Hdf5Writer::fileId, description, matrix.rows(), matrix.cols(), matrix.data(), H5T_NATIVE_DOUBLE);

}

void hops::Hdf5Writer::write(const std::string &description, const Eigen::VectorXd &vector) const {
    writeData(Hdf5Writer::fileId, description, vector.rows(), 1, vector.data(), H5T_NATIVE_DOUBLE);
}
