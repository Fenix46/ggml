add_library(gc_kvcache STATIC
    gc_kvcache.cpp
)

target_include_directories(gc_kvcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(gc_kvcache PRIVATE ${CMAKE_SOURCE_DIR}/src/common)
target_link_libraries(gc_kvcache PUBLIC gc_common)

set_target_properties(gc_kvcache PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
