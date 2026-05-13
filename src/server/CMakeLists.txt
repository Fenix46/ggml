add_library(gc_server STATIC
    gc_server.cpp
    vendor/httplib.cpp
)

target_include_directories(gc_server PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/engine
    ${CMAKE_SOURCE_DIR}/src/scheduler
    ${CMAKE_SOURCE_DIR}/src/kvcache
    ${CMAKE_SOURCE_DIR}/src/chat
)

target_link_libraries(gc_server PUBLIC gc_engine gc_chat)

add_executable(gc_server_main
    gc_server_main.cpp
)

target_include_directories(gc_server_main PRIVATE
    ${CMAKE_SOURCE_DIR}/src/chat
    ${CMAKE_SOURCE_DIR}/src/loader
    ${CMAKE_SOURCE_DIR}/src/arch
    ${CMAKE_SOURCE_DIR}/src/vocab
)

target_link_libraries(gc_server_main PUBLIC
    gc_server
    gc_engine
    gc_loader
    gc_arch
    gc_vocab
    gc_chat
)

set_target_properties(gc_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(gc_server_main PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
