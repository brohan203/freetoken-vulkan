# FreeToken-Vulkan — AMD/Vulkan compute backend
#
# Phase 1: vector_add. The whole pipeline (SPIR-V compile, host code, dispatch,
# readback, verify) as one executable. If this builds and prints PASS on your
# 6800 XT, every downstream kernel we write can lean on the same scaffolding.

cmake_minimum_required(VERSION 3.20)
project(freetoken_vulkan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan SDK provides the loader + headers + glslc.
find_package(Vulkan REQUIRED COMPONENTS glslc)
message(STATUS "Vulkan loader: ${Vulkan_LIBRARY}")
message(STATUS "Vulkan headers: ${Vulkan_INCLUDE_DIR}")
message(STATUS "glslc:         ${Vulkan_GLSLC_EXECUTABLE}")

# ---- helper: compile a .comp shader into a .spv sitting next to the exe ----
function(add_shader TARGET SHADER)
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPV_OUT ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv)
    add_custom_command(
        OUTPUT  ${SPV_OUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${Vulkan_GLSLC_EXECUTABLE}
                --target-env=vulkan1.2
                -O
                -o ${SPV_OUT}
                ${CMAKE_SOURCE_DIR}/${SHADER}
        DEPENDS ${CMAKE_SOURCE_DIR}/${SHADER}
        COMMENT "glslc ${SHADER}"
        VERBATIM
    )
    target_sources(${TARGET} PRIVATE ${SPV_OUT})
endfunction()

# ---- shared include dir ----
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---- vector_add: phase 1 executable ----
add_executable(vector_add src/vector_add.cpp)
target_link_libraries(vector_add PRIVATE Vulkan::Vulkan)
add_shader(vector_add shaders/vector_add.comp)

# ---- rmsnorm: phase 2 executable ----
add_executable(rmsnorm src/rmsnorm.cpp)
target_link_libraries(rmsnorm PRIVATE Vulkan::Vulkan)
add_shader(rmsnorm shaders/rmsnorm_f32.comp)

# ---- gemm: phase 3 executable (shader path passed at runtime) ----
add_executable(gemm src/gemm.cpp)
target_link_libraries(gemm PRIVATE Vulkan::Vulkan)
add_shader(gemm shaders/gemm_naive_f32.comp)
add_shader(gemm shaders/gemm_tiled_f32.comp)
add_shader(gemm shaders/gemm_reg_tiled_f32.comp)
add_shader(gemm shaders/gemm_reg_tiled_f16.comp)

# ---- softmax: phase 4 executable ----
add_executable(softmax src/softmax.cpp)
target_link_libraries(softmax PRIVATE Vulkan::Vulkan)
add_shader(softmax shaders/softmax_f32.comp)

# ---- attention: phase 4 executable (fused single-head) ----
add_executable(attention src/attention.cpp)
target_link_libraries(attention PRIVATE Vulkan::Vulkan)
add_shader(attention shaders/attention_naive_f32.comp)
add_shader(attention shaders/flash_attention_f32.comp)
add_shader(attention shaders/flash_attention_mh_f32.comp)
add_shader(attention shaders/flash_attention_gpt_oss_f32.comp)
add_shader(attention shaders/flash_attention_gpt_oss_kv_f32.comp)
add_shader(attention shaders/flash_attention_gpt_oss_kv_cache_f32.comp)
add_shader(attention shaders/kv_cache_append_f32.comp)
add_shader(attention shaders/topk_softmax_128_f32.comp)
add_shader(attention shaders/transpose_shd_hsd_f32.comp)
add_shader(attention shaders/rope_partial_f32.comp)
add_shader(attention shaders/linear_fp32_resident_f32.comp)
add_shader(attention shaders/linear_fp16_resident_f32.comp)
add_shader(attention shaders/linear_bf16_resident_f32.comp)
add_shader(attention shaders/linear_fp8e4m3_resident_f32.comp)
add_shader(attention shaders/linear_awq4_resident_f32.comp)
add_shader(attention shaders/linear_awq4_stage1_f32.comp)
add_shader(attention shaders/linear_awq4_stage2_f32.comp)

# ---- moe_router: phase 6 executable ----
add_executable(moe_router src/moe_router.cpp)
target_link_libraries(moe_router PRIVATE Vulkan::Vulkan)
add_shader(moe_router shaders/moe_router_f32.comp)

# ---- swiglu: phase 6 executable ----
add_executable(swiglu src/swiglu.cpp)
target_link_libraries(swiglu PRIVATE Vulkan::Vulkan)
add_shader(swiglu shaders/swiglu_f32.comp)

# ---- moe_mlp: phase 6 executable (fused per-token MoE MLP) ----
add_executable(moe_mlp src/moe_mlp.cpp)
target_link_libraries(moe_mlp PRIVATE Vulkan::Vulkan)
add_shader(moe_mlp shaders/moe_mlp_f32.comp)
add_shader(moe_mlp shaders/moe_mlp_lg_f32.comp)

# MXFP4 matvec — pattern-establishing kernel, tested via C++ extension only.
# Compile the SPIR-V but no dedicated CLI target since testing goes through
# the extension.
add_custom_target(mxfp4_matvec_spv ALL
    DEPENDS ${CMAKE_BINARY_DIR}/shaders/mxfp4_matvec_f32.comp.spv
            ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_f32.comp.spv
            ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp.spv
            ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv
            ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/shaders/mxfp4_matvec_f32.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE}
            --target-env=vulkan1.2
            -O
            -o ${CMAKE_BINARY_DIR}/shaders/mxfp4_matvec_f32.comp.spv
            ${CMAKE_SOURCE_DIR}/shaders/mxfp4_matvec_f32.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/mxfp4_matvec_f32.comp
    COMMENT "glslc shaders/mxfp4_matvec_f32.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_f32.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE}
            --target-env=vulkan1.2
            -O
            -o ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_f32.comp.spv
            ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_f32.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_f32.comp
    COMMENT "glslc shaders/moe_mlp_mxfp4_gpt_oss_f32.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE}
            --target-env=vulkan1.2
            -O
            -o ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp.spv
            ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp
    COMMENT "glslc shaders/moe_mlp_mxfp4_gpt_oss_par_f32.comp"
    VERBATIM
)

add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE}
            --target-env=vulkan1.2 -O
            -o ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp.spv
            ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp
    COMMENT "glslc shaders/moe_mlp_mxfp4_gpt_oss_stage1_f32.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT  ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
    COMMAND ${Vulkan_GLSLC_EXECUTABLE}
            --target-env=vulkan1.2 -O
            -o ${CMAKE_BINARY_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp.spv
            ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp
    COMMENT "glslc shaders/moe_mlp_mxfp4_gpt_oss_stage2_f32.comp"
    VERBATIM
)

# MSVC quality-of-life
if(MSVC)
    target_compile_options(vector_add PRIVATE /W4 /permissive-)
    target_compile_options(rmsnorm    PRIVATE /W4 /permissive-)
    target_compile_options(gemm       PRIVATE /W4 /permissive-)
    target_compile_options(softmax    PRIVATE /W4 /permissive-)
    target_compile_options(attention  PRIVATE /W4 /permissive-)
    target_compile_options(moe_router PRIVATE /W4 /permissive-)
    target_compile_options(swiglu     PRIVATE /W4 /permissive-)
    target_compile_options(moe_mlp    PRIVATE /W4 /permissive-)
endif()
