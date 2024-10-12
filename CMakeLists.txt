cmake_minimum_required(VERSION 3.28)
project(BatmanInfer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# 下载GoogleTest项目
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 查找Armadillo库
find_package(Armadillo REQUIRED)

# 查找glog
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLOG REQUIRED libglog)

# 启动测试
enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/includes)
include_directories(${GLOG_INCLUDE_DIRS})

link_directories(${GLOG_LIBRARY_DIRS})

add_executable(BatmanInfer main.cpp
        test/test_operator.cpp
        source/Tensor.cpp
        test/test_create_tensor.cpp
        test/tensor_get_values.cpp)

# 链接GoogleTest库
target_link_libraries(
        BatmanInfer
        GTest::gtest_main
        ${ARMADILLO_LIBRARIES}
        ${GLOG_LIBRARIES}
)

# 添加测试
include(GoogleTest)
gtest_discover_tests(BatmanInfer)

# 包含Armadillo头文件路径
target_include_directories(BatmanInfer PRIVATE ${ARMADILLO_INCLUDE_DIRS})