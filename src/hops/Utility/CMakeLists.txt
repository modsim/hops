if (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
    target_sources(hops PRIVATE
        #ChainData.hpp
        #Data.hpp
        #Data.cpp
        #Exceptions.hpp
        HopsWithinHopsy.hpp
        MatrixType.hpp
        #Problem.hpp
        #Run.hpp
        StringUtility.hpp
        StringUtility.cpp
        VectorType.hpp
    )
endif (NOT HOPS_LIBRARY_TYPE STREQUAL "HEADER_ONLY")
