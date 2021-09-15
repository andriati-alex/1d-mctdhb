cmake_minimum_required(VERSION 3.16)


project(
  1d-mctdhb VERSION 1.0
  DESCRIPTION "Multiconfigurational Time-Dependent Hartree method for Bosons in 1D"
  LANGUAGES C
)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(LIB_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) # To found libraries after target install


add_subdirectory(src)


add_executable(mctdhb_main apps/mctdhb_main.c)
target_link_libraries(mctdhb_main PUBLIC mctdhblib)
set_target_properties(mctdhb_main PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)


install(TARGETS mctdhblib DESTINATION lib)
install(TARGETS mctdhb_main DESTINATION bin)
