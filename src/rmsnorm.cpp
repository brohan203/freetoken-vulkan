// rmsnorm.cpp — CLI wrapper around the RMSNorm compute shader.
//
// Reads raw float32 tensors from disk, dispatches the shader, writes output.
// Python side (tests/test_rmsnorm.py) generates the inputs, computes the
// reference via torch.nn.functional.rms_norm, and diffs the two outputs.
//
// Usage: rmsnorm <x_path> <w_path> <y_path> <N> <H> <eps>
//   x: [N, H] float32     input tensor
//   w: [H]    float32     RMSNorm weight (gamma)
//   y: [N, H] float32     output tensor (this program writes)
//   N: uint32             batch (rows)
//   H: uint32             hidden dim
//   eps: float            RMSNorm epsilon (e.g. 1e-6)

#include "vk_util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void read_bin(const char* path, void* dst, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open input %s\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
    if (!f) { std::fprintf(stderr, "short read from %s\n", path); std::exit(1); }
}

static void write_bin(const char* path, const void* src, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open output %s\n", path); std::exit(1); }
    f.write(reinterpret_cast<const char*>(src), (std::streamsize)bytes);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr,
            "usage: %s <x.bin> <w.bin> <y.bin> <N> <H> <eps>\n", argv[0]);
        return 2;
    }
    const char* x_path = argv[1];
    const char* w_path = argv[2];
    const char* y_path = argv[3];
    const uint32_t N   = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    const uint32_t H   = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    const float    eps = std::strtof(argv[6], nullptr);

    const size_t xy_bytes = (size_t)N * H * sizeof(float);
    const size_t w_bytes  = (size_t)H * sizeof(float);

    std::printf("rmsnorm: N=%u H=%u eps=%.3e  (%.2f MB per xy buffer)\n",
                N, H, eps, xy_bytes / (1024.0 * 1024.0));

    // ---- Vulkan setup (was ~200 lines in vector_add.cpp; now 1 call) ----
    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    // ---- Allocate buffers and upload inputs ----
    vku::Buffer X = vku::make_host_ssbo(ctx, xy_bytes);
    vku::Buffer W = vku::make_host_ssbo(ctx, w_bytes);
    vku::Buffer Y = vku::make_host_ssbo(ctx, xy_bytes);

    std::vector<float> x_host((size_t)N * H);
    std::vector<float> w_host((size_t)H);
    read_bin(x_path, x_host.data(), xy_bytes);
    read_bin(w_path, w_host.data(), w_bytes);
    vku::upload(ctx, X, x_host.data(), xy_bytes);
    vku::upload(ctx, W, w_host.data(), w_bytes);

    // ---- Descriptor set layout: 3 storage buffers, all compute stage ----
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

    // ---- Push constants: struct { uint H; float eps; } ----
    struct PC { uint32_t H; float eps; } pc = { H, eps };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PC);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    // ---- Shader + pipeline ----
    auto spv = vku::load_spirv("shaders/rmsnorm_f32.comp.spv");
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    // ---- Descriptor pool + set ----
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
        {X.buf, 0, VK_WHOLE_SIZE},
        {W.buf, 0, VK_WHOLE_SIZE},
        {Y.buf, 0, VK_WHOLE_SIZE},
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

    // ---- Dispatch: one workgroup per row ----
    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, N, 1, 1);
    });
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    // ---- Read back and write output ----
    std::vector<float> y_host((size_t)N * H);
    vku::download(ctx, Y, y_host.data(), xy_bytes);
    write_bin(y_path, y_host.data(), xy_bytes);

    // ---- Cleanup ----
    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, X);
    vku::destroy_buffer(ctx, W);
    vku::destroy_buffer(ctx, Y);
    vku::destroy_context(ctx);
    return 0;
}
