if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DegenerateGaussian.hpp
            DegenerateGaussian.cpp
            Gaussian.hpp
            Gaussian.cpp
            JumpableModel.hpp
            Model.hpp
            Mixture.hpp
            Rosenbrock.hpp
            Rosenbrock.cpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
