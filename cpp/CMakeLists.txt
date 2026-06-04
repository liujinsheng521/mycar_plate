cmake_minimum_required(VERSION 3.10)

project(rknn_plate_detection_demo)

# Path to rknn_model_zoo root (adjust if needed)
set(RKNN_MODEL_ZOO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../rknn_model_zoo/rknn_model_zoo-2.0.0)

add_subdirectory(${RKNN_MODEL_ZOO_PATH}/3rdparty/ 3rdparty.out)
add_subdirectory(${RKNN_MODEL_ZOO_PATH}/utils/ utils.out)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  set(TARGET_LIB_ARCH aarch64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TARGET_LIB_ARCH lib64)
else()
  set(TARGET_LIB_ARCH lib)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-android-sdk-build/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-osx-x86_64/share/OpenCV)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-windows-x86_64/x64/mingw/staticlib)
else()
  if (TARGET_SOC STREQUAL "x86-64")
    # Native x86-64 build
  elseif(TARGET_LIB_ARCH STREQUAL "aarch64")
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  elseif(TARGET_LIB_ARCH STREQUAL "lib")
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-armhf/share/OpenCV)
  else()
    set(OpenCV_DIR ${RKNN_MODEL_ZOO_PATH}/3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
  endif()
endif()
find_package(OpenCV REQUIRED)

set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

add_executable(${PROJECT_NAME}
    main.cc
    plate_detector.cc
    lprnet.cc
)

target_link_libraries(${PROJECT_NAME}
    fileutils
    imageutils
    imagedrawing
    ${LIBRKNNRT}
    ${OpenCV_LIBS}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME} log)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${RKNN_MODEL_ZOO_PATH}/3rdparty/rknpu2/include
    ${RKNN_MODEL_ZOO_PATH}/3rdparty/librga/include
)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME})
file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../test_data/test.jpg DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/model)

# Install runtime libs
install(FILES ${RKNN_MODEL_ZOO_PATH}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so
        DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/lib)
install(FILES ${RKNN_MODEL_ZOO_PATH}/3rdparty/librga/Linux/aarch64/librga.so
        DESTINATION ${CMAKE_BINARY_DIR}/install/${PROJECT_NAME}/lib)
