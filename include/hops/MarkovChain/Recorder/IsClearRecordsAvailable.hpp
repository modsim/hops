#ifndef HOPS_ISCLEARRECORDSAVAILABLE_HPP
#define HOPS_ISCLEARRECORDSAVAILABLE_HPP

#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsClearRecordsAvailable : std::false_type {
    };

    template<typename T>
    struct IsClearRecordsAvailable<T, std::void_t<decltype(std::declval<T>().clearRecords())> > :
            std::true_type {
    };
}

#endif //HOPS_ISCLEARRECORDSAVAILABLE_HPP
