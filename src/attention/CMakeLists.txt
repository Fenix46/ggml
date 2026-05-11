add_library(gc_attention STATIC
    gc_rope.cpp
    gc_attention.cpp
)

target_include_directories(gc_attention PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(gc_attention PUBLIC ggml)

set_target_properties(gc_attention PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
