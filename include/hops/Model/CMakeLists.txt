if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DegenerateMultivariateGaussian.hpp
            GammaModels.hpp
            Model.hpp
            Mixture.hpp
            MultivariateGaussian.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
