// gemm.cpp — CLI wrapper around a GEMM compute shader.
//
// Takes the SPIR-V path as an argument so we can swap kernel variants
// (naive, tiled, register-tiled...) without rebuilding.
//
// Usage: gemm <shader.spv> <A.bin> <B.bin> <C.bin> <M> <N> <K> [iters]
//   A: [M, K] float32
//   B: [K, N] float32
//   C: [M, N] float32   (this program writes)
//   iters: default 1. If >1, run the kernel that many times and report
//          median dispatch time — useful for benchmarking, because the
//          first dispatch pays pipeline-warmup cost.

#include "vk_util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <fstream>
#include <vector>

static void read_bin(const char* path, void* dst, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
    if (!f) { std::fprintf(stderr, "short read from %s\n", path); std::exit(1); }
}
static void write_bin(const char* path, const void* src, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    f.write(reinterpret_cast<const char*>(src), (std::streamsize)bytes);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: %s <shader.spv> <A.bin> <B.bin> <C.bin> <M> <N> <K> "
            "[iters] [tile_m] [tile_n]\n"
            "  tile_m, tile_n: output tile per workgroup (default 16, 16).\n"
            "  Must match the shader's BM, BN.\n",
            argv[0]);
        return 2;
    }
    const char* spv_path = argv[1];
    const char* a_path   = argv[2];
    const char* b_path   = argv[3];
    const char* c_path   = argv[4];
    const uint32_t M     = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    const uint32_t N     = (uint32_t)std::strtoul(argv[6], nullptr, 10);
    const uint32_t K     = (uint32_t)std::strtoul(argv[7], nullptr, 10);
    const int iters      = (argc > 8)  ? std::atoi(argv[8]) : 1;
    const uint32_t TM    = (argc > 9)  ? (uint32_t)std::atoi(argv[9])  : 16u;
    const uint32_t TN    = (argc > 10) ? (uint32_t)std::atoi(argv[10]) : 16u;

    // Sniff dtype from shader path — F16 shaders use half the bytes for A/B/C.
    const bool is_f16 = (std::string(spv_path).find("_f16") != std::string::npos);
    const size_t elem_bytes = is_f16 ? 2u : 4u;
    const size_t a_bytes = (size_t)M * K * elem_bytes;
    const size_t b_bytes = (size_t)K * N * elem_bytes;
    const size_t c_bytes = (size_t)M * N * elem_bytes;

    std::printf("gemm: shader=%s  M=%u N=%u K=%u  iters=%d  dtype=%s\n",
                spv_path, M, N, K, iters, is_f16 ? "f16" : "f32");

    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    vku::Buffer A = vku::make_host_ssbo(ctx, a_bytes);
    vku::Buffer B = vku::make_host_ssbo(ctx, b_bytes);
    vku::Buffer C = vku::make_host_ssbo(ctx, c_bytes);

    std::vector<char> a_host_raw(a_bytes), b_host_raw(b_bytes);
    read_bin(a_path, a_host_raw.data(), a_bytes);
    read_bin(b_path, b_host_raw.data(), b_bytes);
    vku::upload(ctx, A, a_host_raw.data(), a_bytes);
    vku::upload(ctx, B, b_host_raw.data(), b_bytes);

    // Descriptor set layout: 3 storage buffers.
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 3;
    dsli.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &dsl));

    struct PC { uint32_t M, N, K; } pc = { M, N, K };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(PC);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    auto spv = vku::load_spirv(spv_path);
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));

    VkDescriptorBufferInfo dbi[3] = {
        {A.buf, 0, VK_WHOLE_SIZE},
        {B.buf, 0, VK_WHOLE_SIZE},
        {C.buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    // Dispatch: workgroups cover a tile_m × tile_n block of C each.
    // Convention across all GEMM shaders: local_x → n (column, fast dim),
    // local_y → m (row). So gx tiles N and gy tiles M.
    const uint32_t gx = (N + TN - 1) / TN;
    const uint32_t gy = (M + TM - 1) / TM;

    // Warmup dispatch (not timed) — first launch pays pipeline validation
    // + cache-fill cost. Skew skews benchmarks otherwise.
    (void)vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, gx, gy, 1);
    });

    // Timestamped runs — measures actual GPU compute time separately from
    // queue submission + wait overhead. See vk_util.hpp for details.
    vku::TimestampQuery tsq = vku::make_timestamp_query(ctx);

    std::vector<double> wall_times, gpu_times;
    wall_times.reserve((size_t)iters);
    gpu_times.reserve((size_t)iters);
    for (int i = 0; i < iters; ++i) {
        double wall_ms = vku::submit_with_timestamps(ctx, tsq,
            [&](VkCommandBuffer cbuf) {
                vku::cmd_reset_and_write_start(cbuf, tsq);
                vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        pipeline_layout, 0, 1, &dset, 0, nullptr);
                vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(PC), &pc);
                vkCmdDispatch(cbuf, gx, gy, 1);
                vku::cmd_write_end(cbuf, tsq);
            });
        double gpu_ms = vku::read_gpu_ms(ctx, tsq);
        wall_times.push_back(wall_ms);
        gpu_times.push_back(gpu_ms);
    }

    vku::destroy_timestamp_query(ctx, tsq);

    std::sort(wall_times.begin(), wall_times.end());
    std::sort(gpu_times.begin(), gpu_times.end());
    double median_wall = wall_times[wall_times.size() / 2];
    double median_gpu  = gpu_times [gpu_times.size()  / 2];
    // GEMM does 2*M*N*K flops.
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops_gpu  = flops / (median_gpu  * 1e6);
    double gflops_wall = flops / (median_wall * 1e6);
    std::printf("wall: median=%.3f ms  ->  %.2f GFLOPS\n", median_wall, gflops_wall);
    std::printf("GPU:  median=%.3f ms  ->  %.2f GFLOPS  (real compute time)\n",
                median_gpu, gflops_gpu);

    std::vector<char> c_host_raw(c_bytes);
    vku::download(ctx, C, c_host_raw.data(), c_bytes);
    write_bin(c_path, c_host_raw.data(), c_bytes);

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, A);
    vku::destroy_buffer(ctx, B);
    vku::destroy_buffer(ctx, C);
    vku::destroy_context(ctx);
    return 0;
}
