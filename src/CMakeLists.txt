include_directories( ../external/ )

# Libraries
add_library(densecrf_utils densecrf_utils.cpp)

add_library( densecrf permutohedral.cpp unary.cpp pairwise.cpp densecrf.cpp labelcompatibility.cpp )
target_link_libraries(densecrf densecrf_utils)

