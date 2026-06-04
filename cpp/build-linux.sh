#!/bin/bash

# Build script for RKNN plate detection demo (Rockchip Linux aarch64)
# Usage: ./build-linux.sh [rk3588]

set -e

# Target SOC
TARGET_SOC=${1:-rk3588}

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# RK3588 SDK root (adjust if different)
RK3588_SDK_ROOT="${SCRIPT_DIR}/../../../../../../.."

# Cross-compiler (from SDK prebuilts)
TOOLCHAIN_DIR="${RK3588_SDK_ROOT}/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
CC="${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-gcc"
CXX="${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-g++"

if [ ! -f "$CC" ]; then
    echo "Error: Cross-compiler not found at: $CC"
    echo "Please set TOOLCHAIN_DIR to the correct path."
    exit 1
fi

echo "Using compiler: $($CC --version | head -1)"

# Build directory
BUILD_DIR="build/build_${TARGET_SOC}_linux_aarch64_Release"

echo "Building for ${TARGET_SOC}..."
echo "Build directory: ${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTARGET_SOC=${TARGET_SOC} \
      -DCMAKE_SYSTEM_NAME=Linux \
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
      -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}"

# Build only (skip install to avoid 3rdparty permission issue)
cmake --build "${BUILD_DIR}" -j$(nproc)

# Manually install our project files
INSTALL_DIR="${BUILD_DIR}/install/rknn_plate_detection_demo"
mkdir -p "${INSTALL_DIR}"/{model,lib}

cp "${BUILD_DIR}/rknn_plate_detection_demo" "${INSTALL_DIR}/"

# Copy model and test image
cp "${SCRIPT_DIR}/../model/"*.rknn "${INSTALL_DIR}/model/" 2>/dev/null || true
cp "${SCRIPT_DIR}/../test_data/test.jpg" "${INSTALL_DIR}/model/" 2>/dev/null || true

# Copy runtime libraries
RKNN_ZOO="${SCRIPT_DIR}/../../../../../rknn_model_zoo/rknn_model_zoo-2.0.0"
cp "${RKNN_ZOO}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so" "${INSTALL_DIR}/lib/"
cp "${RKNN_ZOO}/3rdparty/librga/Linux/aarch64/librga.so" "${INSTALL_DIR}/lib/" 2>/dev/null || true

echo ""
echo "Build complete!"
echo "Output: ${INSTALL_DIR}"
echo ""
echo "To run on the board:"
echo "  cd ${INSTALL_DIR}"
echo "./rknn_plate_detection_demo model/bast.rknn model/lprnet.rknn model/test.jpg"
