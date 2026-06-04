#!/bin/bash

# RKNN 车牌检测演示程序 Linux 交叉编译脚本
# 用于在 x86 主机上编译 ARM64 架构的可执行文件
# 用法: ./build-linux.sh [rk3588]

set -e  # 任何命令失败则立即退出脚本

# 目标 SOC 型号，默认 rk3588
TARGET_SOC=${1:-rk3588}

# 获取脚本所在目录的绝对路径
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# RK3588 SDK 根目录（根据实际目录层级计算）
RK3588_SDK_ROOT="${SCRIPT_DIR}/../../../../../../.."

# ARM64 交叉编译器路径（来自 SDK 预编译工具链）
TOOLCHAIN_DIR="${RK3588_SDK_ROOT}/prebuilts/gcc/linux-x86/aarch64/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
CC="${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-gcc"    # C 编译器
CXX="${TOOLCHAIN_DIR}/bin/aarch64-none-linux-gnu-g++"   # C++ 编译器

# 检查交叉编译器是否存在
if [ ! -f "$CC" ]; then
    echo "Error: Cross-compiler not found at: $CC"
    echo "Please set TOOLCHAIN_DIR to the correct path."
    exit 1
fi

echo "Using compiler: $($CC --version | head -1)"  # 打印编译器版本信息

# 构建输出目录（按 SOC 和平台分类）
BUILD_DIR="build/build_${TARGET_SOC}_linux_aarch64_Release"

echo "Building for ${TARGET_SOC}..."
echo "Build directory: ${BUILD_DIR}"

# 执行 CMake 配置，生成 Makefile
cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \         # Release 模式（开启优化）
      -DTARGET_SOC=${TARGET_SOC} \          # 目标 SOC 型号
      -DCMAKE_SYSTEM_NAME=Linux \           # 目标系统为 Linux
      -DCMAKE_SYSTEM_PROCESSOR=aarch64 \    # 目标架构为 ARM64
      -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \  # 安装路径前缀
      -DCMAKE_C_COMPILER="${CC}" \          # 指定 C 交叉编译器
      -DCMAKE_CXX_COMPILER="${CXX}"         # 指定 C++ 交叉编译器

# 执行编译（仅编译不安装，避免 3rdparty 目录权限问题）
cmake --build "${BUILD_DIR}" -j$(nproc)     # -j$(nproc) 并行编译，使用所有 CPU 核心

# ====== 手动安装项目文件到输出目录 ======
INSTALL_DIR="${BUILD_DIR}/install/rknn_plate_detection_demo"
mkdir -p "${INSTALL_DIR}"/{model,lib}       # 创建 model 和 lib 子目录

# 复制编译好的可执行文件
cp "${BUILD_DIR}/rknn_plate_detection_demo" "${INSTALL_DIR}/"

# 复制 RKNN 模型文件（如果存在）
cp "${SCRIPT_DIR}/../model/"*.rknn "${INSTALL_DIR}/model/" 2>/dev/null || true
# 复制测试图片（如果存在）
cp "${SCRIPT_DIR}/../test_data/test.jpg" "${INSTALL_DIR}/model/" 2>/dev/null || true

# 复制 NPU 运行时动态库
RKNN_ZOO="${SCRIPT_DIR}/../../../../../rknn_model_zoo/rknn_model_zoo-2.0.0"
cp "${RKNN_ZOO}/3rdparty/rknpu2/Linux/aarch64/librknnrt.so" "${INSTALL_DIR}/lib/"
# 复制 RGA 图像加速库（如果存在）
cp "${RKNN_ZOO}/3rdparty/librga/Linux/aarch64/librga.so" "${INSTALL_DIR}/lib/" 2>/dev/null || true

echo ""
echo "Build complete!"
echo "Output: ${INSTALL_DIR}"
echo ""
echo "To run on the board:"
echo "  cd ${INSTALL_DIR}"
echo "./rknn_plate_detection_demo model/bast.rknn model/lprnet.rknn model/test.jpg"
