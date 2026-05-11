add_library(gc_loader STATIC
    gc_common.cpp
    gc_mmap.cpp
    gc_gguf_loader.cpp
)

target_include_directories(gc_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include      # ggml.h, gguf.h, etc.
)

target_link_libraries(gc_loader PRIVATE ggml)

set_target_properties(gc_loader PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
