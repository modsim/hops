#ifndef NUPS_STATERECORDER_HPP
#define NUPS_STATERECORDER_HPP

#include <nups/FileWriter/FileWriter.hpp>
#include <nups/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <nups/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <vector>

namespace nups {
    template<typename MarkovChainImpl>
    class StateRecorder : public MarkovChainImpl {
    public:
        explicit StateRecorder(const MarkovChainImpl &markovChainImpl) : MarkovChainImpl(markovChainImpl) {}

//        void draw(RandomNumberGenerator &randomNumberGenerator) {
//            MarkovChainImpl::draw(randomNumberGenerator);
//        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("states", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        void storeRecord() {
            records.emplace_back(MarkovChainImpl::getState());
            if constexpr(IsStoreRecordAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::storeRecord();
            }
        }

        void clearRecords() {
            records.clear();
            if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                records.clear();
                if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                    MarkovChainImpl::clearRecords();
                }
            }
        }

    private:
        std::vector<typename MarkovChainImpl::StateType> records;
    };

}

#endif //NUPS_STATERECORDER_HPP
