if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DegenerateGaussian.hpp
            GammaModels.hpp
            Model.hpp
            Mixture.hpp
            Gaussian.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
