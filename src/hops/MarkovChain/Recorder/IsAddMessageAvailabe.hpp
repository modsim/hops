#ifndef HOPS_ISADDMESSAGEAVAILABE_HPP
#define HOPS_ISADDMESSAGEAVAILABE_HPP

#include <string>
#include <type_traits>

namespace hops {
    template<typename T, typename = void>
    struct IsAddMessageAvailable : std::false_type {
    };

    template<typename T>
    struct IsAddMessageAvailable<T, std::void_t<decltype(std::declval<T>().addMessage(
            std::declval<const std::string &>()))> > :
            std::true_type {
    };
}

#endif //HOPS_ISADDMESSAGEAVAILABE_HPP
