#ifndef HOPS_ISSETFISHERWEIGHTAVAILABLE_HPP
#define HOPS_ISSETFISHERWEIGHTAVAILABLE_HPP


namespace hops {
    template<typename T, typename = void>
    struct IsSetFisherWeightAvailable : std::false_type {
    };

    template<typename T>
    struct IsSetFisherWeightAvailable<T,
            std::void_t < decltype(std::declval<T>().setFisherWeight(std::declval<double>()))> > :
    std::true_type {
};
}

#endif //HOPS_ISSETFISHERWEIGHTAVAILABLE_HPP
