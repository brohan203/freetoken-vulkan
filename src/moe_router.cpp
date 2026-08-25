// moe_router.cpp — CLI wrapper for the top-K MoE router.
//
// Usage: moe_router <logits.bin> <indices.bin> <weights.bin> <T> <E> <K>
//   logits:  [T, E] float32
//   indices: [T, K] uint32   (this program writes)
//   weights: [T, K] float32  (this program writes)

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
    if (argc != 7) {
        std::fprintf(stderr,
            "usage: %s <logits.bin> <indices.bin> <weights.bin> <T> <E> <K>\n",
            argv[0]);
        return 2;
    }
    const char* logits_path  = argv[1];
    const char* indices_path = argv[2];
    const char* weights_path = argv[3];
    const uint32_t T = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    const uint32_t E = (uint32_t)std::strtoul(argv[5], nullptr, 10);
    const uint32_t K = (uint32_t)std::strtoul(argv[6], nullptr, 10);

    const size_t logits_bytes  = (size_t)T * E * sizeof(float);
    const size_t indices_bytes = (size_t)T * K * sizeof(uint32_t);
    const size_t weights_bytes = (size_t)T * K * sizeof(float);

    std::printf("moe_router: T=%u E=%u K=%u\n", T, E, K);

    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    vku::Buffer L = vku::make_host_ssbo(ctx, logits_bytes);
    vku::Buffer I = vku::make_host_ssbo(ctx, indices_bytes);
    vku::Buffer W = vku::make_host_ssbo(ctx, weights_bytes);
    std::vector<float> l_host((size_t)T * E);
    read_bin(logits_path, l_host.data(), logits_bytes);
    vku::upload(ctx, L, l_host.data(), logits_bytes);

    VkDescriptorSetLayoutBinding bindings[3]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 3; dsli.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &dsl));

    struct PC { uint32_t E; uint32_t K; } pc = { E, K };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(PC);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    auto spv = vku::load_spirv("shaders/moe_router_f32.comp.spv");
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));

    VkDescriptorBufferInfo dbi[3] = {
        {L.buf, 0, VK_WHOLE_SIZE},
        {I.buf, 0, VK_WHOLE_SIZE},
        {W.buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);

    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, T, 1, 1);
    });
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    std::vector<uint32_t> i_host((size_t)T * K);
    std::vector<float>    w_host((size_t)T * K);
    vku::download(ctx, I, i_host.data(), indices_bytes);
    vku::download(ctx, W, w_host.data(), weights_bytes);
    write_bin(indices_path, i_host.data(), indices_bytes);
    write_bin(weights_path, w_host.data(), weights_bytes);

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, L);
    vku::destroy_buffer(ctx, I);
    vku::destroy_buffer(ctx, W);
    vku::destroy_context(ctx);
    return 0;
}
