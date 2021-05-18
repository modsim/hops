if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            DegenerateMultivariateGaussianModel.hpp
            DynMultimodalModel.hpp
            ModelMixin.hpp
            MultimodalModel.hpp
            MultivariateGaussianModel.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
