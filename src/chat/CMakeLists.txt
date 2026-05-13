add_library(gc_chat STATIC gc_chat.cpp)

target_include_directories(gc_chat PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(gc_chat PUBLIC gc_common)

set_target_properties(gc_chat PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
