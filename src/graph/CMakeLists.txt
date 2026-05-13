add_library(gc_graph STATIC
    gc_graph_llama.cpp
)

target_include_directories(gc_graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/arch
    ${CMAKE_SOURCE_DIR}/src/attention
    ${CMAKE_SOURCE_DIR}/src/kvcache
    ${CMAKE_SOURCE_DIR}/src/loader
)

target_link_libraries(gc_graph PUBLIC gc_engine gc_arch gc_attention gc_kvcache gc_loader ggml)

set_target_properties(gc_graph PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
