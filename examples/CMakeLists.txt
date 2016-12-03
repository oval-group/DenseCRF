include_directories(../external/ProbImage ../external/ )

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_library(color_to_label color_to_label.cpp)

add_library(file_storage file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

# Executables
add_executable(inference inference.cpp)
target_link_libraries(inference densecrf file_storage densecrf_utils)

