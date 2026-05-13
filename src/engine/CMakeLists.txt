add_library(gc_engine STATIC
    gc_engine.cpp
    gc_default_model_runner.cpp
    gc_ggml_model_runner.cpp
    gc_graph_runner.cpp
)

target_include_directories(gc_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/graph
)

target_link_libraries(gc_engine PUBLIC gc_scheduler gc_kvcache gc_loader gc_arch gc_attention gc_graph gc_common ggml)

set_target_properties(gc_engine PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
