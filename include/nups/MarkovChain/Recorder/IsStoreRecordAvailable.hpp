#ifndef NUPS_ISSTORERECORDAVAILABLE_HPP
#define NUPS_ISSTORERECORDAVAILABLE_HPP

#include <type_traits>

namespace nups {
    template<typename T, typename = void>
    struct IsStoreRecordAvailable : std::false_type {
    };

    template<typename T>
    struct IsStoreRecordAvailable<T, std::void_t<decltype(std::declval<T>().storeRecord())> > :
            std::true_type {
    };
}

#endif //NUPS_ISSTORERECORDAVAILABLE_HPP
