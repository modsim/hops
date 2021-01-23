# - Try to find SBML

if (SBML_INCLUDE_DIR)
    # in cache already
    set(SBML_FOUND TRUE)
    set(SBML_INCLUDE_DIRS "${SBML_INCLUDE_DIR}" )
    set(SBML_LIBRARIES "${SBML_CXX_LIBRARY};${SBML_LIBRARY}" )
else (SBML_INCLUDE_DIR)

    find_path(SBML_INCLUDE_DIR
            NAMES
            SBMLReader.h
            PATHS
            "/usr/include/sbml"
            "/usr/include/"
            "/usr/local/include/sbml"
            "/usr/local/include/"
            )

    find_library(SBML_LIBRARY
            NAMES sbml
            PATHS
            "/usr/lib/x86_64-linux-gnu/"
            "/usr/lib"
            )

    set(SBML_INCLUDE_DIRS "${SBML_INCLUDE_DIR}" )
    set(SBML_LIBRARIES "${SBML_LIBRARY}" )

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set LIBCPLEX_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(SBML  DEFAULT_MSG
            SBML_LIBRARY SBML_INCLUDE_DIR)

    mark_as_advanced(SBML_INCLUDE_DIR SBML_LIBRARY SBML_CXX_LIBRARY SBML_BIN_DIR )

endif(SBML_INCLUDE_DIR)

