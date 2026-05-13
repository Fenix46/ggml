add_library(gc_scheduler STATIC
    gc_request.cpp
    gc_scheduler.cpp
)

target_include_directories(gc_scheduler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/kvcache
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(gc_scheduler PUBLIC gc_kvcache)

set_target_properties(gc_scheduler PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
