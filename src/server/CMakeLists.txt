add_library(gc_server STATIC
    gc_server.cpp
    vendor/httplib.cpp
)

target_include_directories(gc_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/scheduler
    ${CMAKE_SOURCE_DIR}/src/kvcache
)

target_link_libraries(gc_server PUBLIC gc_engine)

set_target_properties(gc_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
