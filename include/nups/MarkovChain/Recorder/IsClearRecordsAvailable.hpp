#ifndef NUPS_ISCLEARRECORDSAVAILABLE_HPP
#define NUPS_ISCLEARRECORDSAVAILABLE_HPP

#include <type_traits>

namespace nups {
    template<typename T, typename = void>
    struct IsClearRecordsAvailable : std::false_type {
    };

    template<typename T>
    struct IsClearRecordsAvailable<T, std::void_t<decltype(std::declval<T>().clearRecords())> > :
            std::true_type {
    };
}

#endif //NUPS_ISCLEARRECORDSAVAILABLE_HPP
