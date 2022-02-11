if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DegenerateGaussian.hpp
            Gamma.hpp
            Gaussian.hpp
            Model.hpp
            Mixture.hpp
            Rosenbrock.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
