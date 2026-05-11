add_library(gc_engine STATIC gc_engine.cpp)

target_include_directories(gc_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(gc_engine PUBLIC gc_scheduler gc_kvcache)

set_target_properties(gc_engine PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
