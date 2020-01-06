#ifndef NUPS_ISACCEPTPROPOSALAVAILABLE_HPP
#define NUPS_ISACCEPTPROPOSALAVAILABLE_HPP

#include <type_traits>

namespace nups {
    template<typename T, typename = void>
    struct IsAcceptProposalAvailable : std::false_type {
    };

    template<typename T>
    struct IsAcceptProposalAvailable<T, std::void_t<decltype(std::declval<T>().acceptProposal())> > :
            std::true_type {
    };
}

#endif //NUPS_ISACCEPTPROPOSALAVAILABLE_HPP
