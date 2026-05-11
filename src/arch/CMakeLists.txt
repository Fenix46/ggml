add_library(gc_arch STATIC
    gc_arch.cpp
    gc_hparams.cpp
    gc_arch_llama.cpp
    gc_arch_runtime.cpp
    gc_arch_gemma_common.cpp
    gc_arch_gemma.cpp
    gc_arch_gemma2.cpp
    gc_arch_gemma3.cpp
    gc_arch_gemma3n.cpp
    gc_arch_gemma4.cpp
    gc_arch_gemma_embedding.cpp
)

target_include_directories(gc_arch PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/loader
)

target_link_libraries(gc_arch PUBLIC gc_loader ggml)

set_target_properties(gc_arch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
