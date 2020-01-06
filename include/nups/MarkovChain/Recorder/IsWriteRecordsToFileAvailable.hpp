#ifndef NUPS_ISWRITERECORDSTOFILEAVAILABLE_HPP
#define NUPS_ISWRITERECORDSTOFILEAVAILABLE_HPP

#include <nups/FileWriter/FileWriter.hpp>
#include <type_traits>

namespace nups {
    template<typename T, typename = void>
    struct IsWriteRecordsToFileAvailable : std::false_type {
    };

    template<typename T>
    struct IsWriteRecordsToFileAvailable<T,
            std::void_t<decltype(std::declval<T>().writeRecordsToFile(std::declval<const FileWriter *>()))> > :
            std::true_type {
    };
}

#endif //NUPS_ISWRITERECORDSTOFILEAVAILABLE_HPP
