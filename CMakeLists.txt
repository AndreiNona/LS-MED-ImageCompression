cmake_minimum_required(VERSION 3.25)
project(Byte2BitProject1)

set(CMAKE_CXX_STANDARD 20)

# ---- Fetch GoogleTest ----
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---- Main executable ----
add_executable(Byte2BitProject1
        main.cpp
        imageIO.cpp
        imageIO.h
        predictor.cpp
        predictor.h
        third_party/stb/stb_image.h
        third_party/stb/stb_image_write.h
        residualIO.cpp
        residualIO.h
)

target_include_directories(Byte2BitProject1 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

# ---- Test executable ----
add_executable(Byte2BitTests
        tests/predictor_tests.cpp
        imageIO.cpp
        imageIO.h
        predictor.cpp
        predictor.h
        third_party/stb/stb_image.h
        third_party/stb/stb_image_write.h
        tests/test_imageIO.cpp
        tests/test_imageIO.h
)

target_include_directories(Byte2BitTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
)

target_link_libraries(Byte2BitTests PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(Byte2BitTests)
