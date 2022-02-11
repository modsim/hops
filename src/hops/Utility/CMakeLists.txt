if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
            #ChainData.hpp
            #Data.hpp
            #Data.cpp
            #Exceptions.hpp
            HopsWithinHopsy.hpp
            KahanSum.hpp
            KahanSum.cpp
            MatrixType.hpp
            #Problem.hpp
            #Run.hpp
            Sampling.hpp
            Sampling.cpp
            StringUtility.hpp
            StringUtility.cpp
            VectorType.hpp
            )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
