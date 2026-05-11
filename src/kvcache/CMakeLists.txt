add_library(gc_kvcache STATIC
    gc_kvcache.cpp
)

target_include_directories(gc_kvcache PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(gc_kvcache PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
