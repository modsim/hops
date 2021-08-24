#ifndef HOPS_NESTEDSAMPLERFACTORY_HPP
#define HOPS_NESTEDSAMPLERFACTORY_HPP

#include <dnest4/DNest4.h>
#include <hops/NestedSampling/DNest4Adapter.hpp>

namespace hops {
    using DNest4CompatibleModel =
            DNest4DAdapter<hops::MarkovChain,
            hops::MarkovChain,
            model>

   class NestedSamplingFactory {
   public:


       template <typename PriorSampler, typename PosteriorProposer, typename Model>
       static DNest4::Sampler<hops::DNest4Adapter<typename PriorSampler, typename PosteriorProposer, typename Model>>
       createNestedSampler(MarkovChainType PriorSamplerType,
              MarkovChainType PosteriorProposerType,
              const Model &model);

       }

    template<typename PriorSampler, typename PosteriorProposer, typename Model>
    DNest4::Sampler<hops::DNest4Adapter<PriorSampler, PosteriorProposer, Model>>
    NestedSamplingFactory::createNestedSampler(MarkovChainType PriorSamplerType, MarkovChainType PosteriorProposerType,
                                               const Model &model) {

       DNest4::Sampler(

               )

        return nullptr;
    }
};
}

#endif //HOPS_NESTEDSAMPLERFACTORY_HPP
