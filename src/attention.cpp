// attention.cpp — CLI wrapper for the fused attention kernel.
//
// Usage: attention <Q.bin> <K.bin> <V.bin> <O.bin> <S> <D> <scale>
//   Q, K, V, O: [S, D] float32
//   scale: typically 1/sqrt(D). Passed explicitly so tests can experiment.

#include "vk_util.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void read_bin(const char* path, void* dst, size_t bytes) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    f.read(reinterpret_cast<char*>(dst), (std::streamsize)bytes);
}
static void write_bin(const char* path, const void* src, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    f.write(reinterpret_cast<const char*>(src), (std::streamsize)bytes);
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::fprintf(stderr,
            "usage: %s <shader.spv> <Q.bin> <K.bin> <V.bin> <O.bin> <S> <D> <scale>\n",
            argv[0]);
        return 2;
    }
    const char* spv_path = argv[1];
    const char* q_path = argv[2];
    const char* k_path = argv[3];
    const char* v_path = argv[4];
    const char* o_path = argv[5];
    const uint32_t S   = (uint32_t)std::strtoul(argv[6], nullptr, 10);
    const uint32_t D   = (uint32_t)std::strtoul(argv[7], nullptr, 10);
    const float   scale = std::strtof(argv[8], nullptr);
    const size_t bytes = (size_t)S * D * sizeof(float);

    std::printf("attention: shader=%s  S=%u D=%u scale=%.6f\n",
                spv_path, S, D, scale);

    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    vku::Buffer Q = vku::make_host_ssbo(ctx, bytes);
    vku::Buffer K = vku::make_host_ssbo(ctx, bytes);
    vku::Buffer V = vku::make_host_ssbo(ctx, bytes);
    vku::Buffer O = vku::make_host_ssbo(ctx, bytes);
    std::vector<float> q_host((size_t)S * D), k_host((size_t)S * D), v_host((size_t)S * D);
    read_bin(q_path, q_host.data(), bytes);
    read_bin(k_path, k_host.data(), bytes);
    read_bin(v_path, v_host.data(), bytes);
    vku::upload(ctx, Q, q_host.data(), bytes);
    vku::upload(ctx, K, k_host.data(), bytes);
    vku::upload(ctx, V, v_host.data(), bytes);

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (int i = 0; i < 4; ++i) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 4; dsli.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &dsl));

    struct PC { uint32_t S; uint32_t D; float scale; } pc = { S, D, scale };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(PC);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    auto spv = vku::load_spirv(spv_path);
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));

    VkDescriptorBufferInfo dbi[4] = {
        {Q.buf, 0, VK_WHOLE_SIZE}, {K.buf, 0, VK_WHOLE_SIZE},
        {V.buf, 0, VK_WHOLE_SIZE}, {O.buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);

    // Warmup
    (void)vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, S, 1, 1);
    });

    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, S, 1, 1);
    });
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    std::vector<float> o_host((size_t)S * D);
    vku::download(ctx, O, o_host.data(), bytes);
    write_bin(o_path, o_host.data(), bytes);

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, Q);
    vku::destroy_buffer(ctx, K);
    vku::destroy_buffer(ctx, V);
    vku::destroy_buffer(ctx, O);
    vku::destroy_context(ctx);
    return 0;
}
