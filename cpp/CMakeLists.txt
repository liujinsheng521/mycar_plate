cmake_minimum_required(VERSION 3.10)  # 最低 CMake 版本要求

project(rknn_plate_detection_demo)  # 项目名称，也是最终可执行文件名

# rknn_model_zoo 根目录路径（根据实际位置调整）
set(RKNN_MODEL_ZOO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../rknn_model_zoo/rknn_model_zoo-2.0.0)

# 添加第三方依赖库子目录（rknn_api、librga 等）
add_subdirectory(${RKNN_MODEL_ZOO_PATH}/3rdparty/ 3rdparty.out)
# 添加 rknn_model_zoo 工具库子目录（文件读取、图像处理等）
add_subdirectory(${RKNN_MODEL_ZOO_PATH}/utils/ utils.out)

# 根据目标平台架构选择库文件架构
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(TARGET_LIB_ARCH aarch64)       # ARM 64 位
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TARGET_LIB_ARCH lib64)         # x86 64 位
else()
  set(TARGET_LIB_ARCH lib)           # 32 位
endif()

# 根据操作系统选择 OpenCV 路径
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  # Android NDK 交叉编译的 OpenCV
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-osx-x86_64/share/OpenCV)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-windows-x86_64/x64/mingw/staticlib)
else()
  # Linux 平台
  if (TARGET_SOC STREQUAL "x86-64")
    # 本机 x86-64 编译，使用系统安装的 OpenCV
  elseif(TARGET_LIB_ARCH STREQUAL "aarch64")
    # ARM64 交叉编译，使用预编译的 OpenCV
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  elseif(TARGET_LIB_ARCH STREQUAL "lib")
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    # 默认使用 aarch64 OpenCV
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  endif()
endif()
find_package(OpenCV REQUIRED)  # 查找 OpenCV 库

# 设置可执行文件的运行时库搜索路径（rpath），优先在 ../lib 下找 .so
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

# 指定生成可执行文件所需的源文件
add_executable(${PROJECT_NAME}
    main.cc               # 主程序入口
    plate_detector.cc     # YOLOv5 车牌检测实现
    lprnet.cc             # LPRNet 车牌识别实现
)

# 链接所需的库
target_link_libraries(${PROJECT_NAME}
    fileutils      # rknn_model_zoo 文件工具库
    imageutils     # rknn_model_zoo 图像处理库
    imagedrawing   # rknn_model_zoo 图像绘制库
    ${LIBRKNNRT}   # RKNN 运行时库 librknnrt.so
    ${OpenCV_LIBS} # OpenCV 库（core/imgproc/imgcodecs 等）
)

# Android 平台需要链接 log 库
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

# Linux 平台需要链接 pthread 线程库
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# 设置头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}                         # 当前源码目录
    ${LIBRKNNRT_INCLUDES}                               # RKNN API 头文件
    ${RKNN_MODEL_ZOO_PATH}/3rdparty/rknpu2/include      # rknpu2 头文件
    ${RKNN_MODEL_ZOO_PATH}/3rdparty/librga/include       # RGA 图像加速库头文件
)

# 安装配置：将可执行文件安装到构建目录下的 install 子目录
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME})
# 复制 RKNN 模型文件
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/model)
# 复制测试图片
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../test_data/test.jpg DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/model)

# 安装运行时动态库
install(FILES ${RKNN_MODEL_ZOO_PATH}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so
        DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/lib)
install(FILES ${RKNN_MODEL_ZOO_PATH}/3rdparty/librga/Linux/aarch64/librga.so
        DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/lib)
