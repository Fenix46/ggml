add_library(gc_server STATIC
    gc_server.cpp
    vendor/httplib.cpp
)

target_include_directories(gc_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(gc_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
