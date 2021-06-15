#ifndef HOPS_ISCALCULATEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP
#define HOPS_ISCALCULATEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsCalculateExpectedFisherInformationAvailable : std::false_type {
    };

    template<typename T>
    struct IsCalculateExpectedFisherInformationAvailable<T, std::void_t<decltype(std::declval<T>()
            .calculateExpectedFisherInformation(std::declval<const typename T::VectorType &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISCALCULATEEXPECTEDFISHERINFORMATIONAVAILABLE_HPP
