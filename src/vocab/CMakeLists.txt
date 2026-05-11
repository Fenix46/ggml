add_library(gc_vocab STATIC
    gc_unicode.cpp
    gc_unicode_data.cpp
    gc_vocab.cpp
)

target_include_directories(gc_vocab PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/loader
    ${CMAKE_SOURCE_DIR}/src/arch
)

target_link_libraries(gc_vocab PUBLIC gc_arch gc_loader ggml)

set_target_properties(gc_vocab PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
