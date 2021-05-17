#ifndef HOPS_METROPOLISHASTINGSINFORECORDER_HPP
#define HOPS_METROPOLISHASTINGSINFORECORDER_HPP

#include <hops/FileWriter/FileWriter.hpp>
#include <hops/MarkovChain/Recorder/IsClearRecordsAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsInstallDataObjectAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsStoreRecordAvailable.hpp>
#include <hops/MarkovChain/Recorder/IsWriteRecordsToFileAvailable.hpp>
#include <hops/Utility/Data.hpp>

#include <vector>

namespace hops {
    template<typename MarkovChainImpl>
    class MetropolisHastingsInfoRecorder : public MarkovChainImpl {
    public:
        explicit MetropolisHastingsInfoRecorder(const MarkovChainImpl &markovChainImpl) :
                MarkovChainImpl(markovChainImpl) {}

        void installDataObject(ChainData& chainData) {
            if constexpr(IsInstallDataObjectAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::installDataObject(chainData);
            }
        }

        void writeRecordsToFile(const FileWriter *const fileWriter) const {
            fileWriter->write("MetropolisFilter", records);
            if constexpr(IsWriteRecordsToFileAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::writeRecordsToFile(fileWriter);
            }
        };

        void storeMetropolisHastingsInfoRecord(const std::string &info) {
            records.emplace_back(info);
        }

        void appendToLatestMetropolisHastingsInfoRecord(const std::string &infoToAppend) {
            records.back().append(infoToAppend);
        }

        void clearRecords() {
            records.clear();
            if constexpr(IsClearRecordsAvailable<MarkovChainImpl>::value) {
                MarkovChainImpl::clearRecords();
            }
        }

    private:
        std::vector<std::string> records;
    };
}

#endif //HOPS_METROPOLISHASTINGSINFORECORDER_HPP
