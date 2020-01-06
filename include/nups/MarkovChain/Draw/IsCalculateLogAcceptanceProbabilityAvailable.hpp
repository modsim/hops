#ifndef NUPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP
#define NUPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP

#include <type_traits>

namespace nups {
    template<typename T, typename = void>
    struct IsCalculateLogAcceptanceProbabilityAvailable : std::false_type {
    };

    template<typename T>
    struct IsCalculateLogAcceptanceProbabilityAvailable<T, std::void_t<decltype(std::declval<T>().calculateLogAcceptanceProbability())> > :
            std::true_type {
    };
}

#endif //NUPS_ISCALCULATELOGACCEPTANCEPROBABILITYAVAILABLE_HPP
