configure_file(A_small.csv ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(A_small_with_column_and_row_names.csv ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(b_small.csv ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(b_small_with_row_names.csv ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)

add_subdirectory(e_coli_core)
add_subdirectory(hdf5)

