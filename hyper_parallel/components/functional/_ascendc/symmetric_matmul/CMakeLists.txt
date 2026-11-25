# Copyright 2026 Huawei Technologies Co., Ltd
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.16)
project(hp_symm_pipeline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit")
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(hp_pipeline_kernel SHARED fused.cpp)
ascendc_include_directories(hp_pipeline_kernel PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_library(hp_symm_torch SHARED plan.cpp torch_op.cpp)
target_compile_features(hp_symm_torch PRIVATE cxx_std_20)
target_compile_definitions(hp_symm_torch PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
target_include_directories(hp_symm_torch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/.. ${TORCH_DIR}/include ${TORCH_DIR}/include/torch/csrc/api/include
    ${TORCH_NPU_DIR}/include ${PYTHON_INCLUDE} ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment ${ASCEND_CANN_PACKAGE_PATH}/asc/include)
target_link_directories(hp_symm_torch PRIVATE ${TORCH_DIR}/lib ${TORCH_NPU_DIR}/lib ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(hp_symm_torch PRIVATE hp_pipeline_kernel torch_npu torch_cpu c10 tiling_api platform ascendcl)
set_target_properties(hp_symm_torch PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
install(TARGETS hp_symm_torch hp_pipeline_kernel LIBRARY DESTINATION . COMPONENT hp_runtime)
