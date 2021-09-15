set(MKL_LINK static)
set(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL REQUIRED)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE LIB_SOURCES ${LIB_SOURCE_DIR}/*.c)
add_library(mctdhblib SHARED ${LIB_SOURCES})

target_include_directories(mctdhblib PUBLIC ${CMAKE_SOURCE_DIR}/include ~/.local/include)
target_link_directories(mctdhblib PUBLIC /home/andriati/.local/lib)
# target_link_libraries(mctdhblib OpenMP::OpenMP_C)
target_link_libraries(mctdhblib m OpenMP::OpenMP_C mkl_intel_lp64 mkl_gnu_thread mkl_core cpydataio odesys)
set_target_properties(mctdhblib PROPERTIES OUTPUT_NAME mctdhb)
