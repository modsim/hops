# - Try to find DNest4

if (DNest4_INCLUDE_DIR)
    # in cache already
    set(DNest4_FOUND TRUE)
    set(DNest4_INCLUDE_DIRS "${DNest4_INCLUDE_DIR}" )
    set(DNest4_LIBRARIES "${DNest4_LIBRARY}" )
else (DNest4_INCLUDE_DIR)

    find_path(DNest4_INCLUDE_DIR
            NAMES
            DNest4.h
            PATHS
            "/usr/local/include/dnest4"
            )

    find_library(DNest4_LIBRARY
            NAMES dnest4
            PATHS
            "/usr/local/lib"
            )

    set(DNest4_INCLUDE_DIRS "${DNest4_INCLUDE_DIR}" )
    set(DNest4_LIBRARIES "${DNest4_LIBRARY}" )

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(DNest4  DEFAULT_MSG
            DNest4_LIBRARY DNest4_INCLUDE_DIR)

    mark_as_advanced(DNest4_INCLUDE_DIR DNest4_LIBRARY DNest4_CXX_LIBRARY DNest4_BIN_DIR )

endif(DNest4_INCLUDE_DIR)

