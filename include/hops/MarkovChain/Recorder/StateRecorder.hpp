#ifndef HOPS_STATERECORDER_HPP
#define HOPS_STATERECORDER_HPP

#include <memory>
#include <vector>

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsInstallDataObjectAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <hops/Utility/ChainData.hpp>

namespace hops {
    template<typename MarkovChainImpl>
    class StateRecorder : public MarkovChainImpl {
    public:
        explicit StateRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {
            records = std::make_shared<std::vector<typename MarkovChainImpl::StateType>>();
        }

        void installDataObject(ChainData& chainData) {
            chainData.setStates(records);
            if constexpr(IsInstallDataObjectAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::installDataObject(chainData);
            }
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("states", *records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        }

        const std::vector<typename MarkovChainImpl::StateType>& getStateRecords() const {
            return *records;
        }

        void reserveStateRecords(long numberOfSamples) {
            records->reserve(numberOfSamples);
        }

        void storeRecord() {
            records->emplace_back(MarkovChainImpl::getState());
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::storeRecord();
            }
        }

        void clearRecords() {
            records->clear();
            if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::clearRecords();
            }
        }

    private:
        std::shared_ptr<std::vector<typename MarkovChainImpl::StateType>> records;
    };

}

#endif //HOPS_STATERECORDER_HPP
