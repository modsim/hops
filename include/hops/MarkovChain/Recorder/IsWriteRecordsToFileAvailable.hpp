#ifndef HOPS_ISWRITERECORDSTOFILEAVAILABLE_HPP
#define HOPS_ISWRITERECORDSTOFILEAVAILABLE_HPP

#include "../../FileWriter/FileWriter.hpp"
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsWriteRecordsToFileAvailable : std::false_type {
    };

    template<typename T>
    struct IsWriteRecordsToFileAvailable<T,
            std::void_t<decltype(std::declval<T>().writeRecordsToFile(std::declval<const FileWriter *>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISWRITERECORDSTOFILEAVAILABLE_HPP
