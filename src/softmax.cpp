// softmax.cpp — CLI wrapper for the rowwise softmax kernel.
//
// Usage: softmax <x.bin> <y.bin> <N> <D>
//   x: [N, D] float32   input
//   y: [N, D] float32   output (softmax over last dim)

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
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <x.bin> <y.bin> <N> <D>\n", argv[0]);
        return 2;
    }
    const char* x_path = argv[1];
    const char* y_path = argv[2];
    const uint32_t N = (uint32_t)std::strtoul(argv[3], nullptr, 10);
    const uint32_t D = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    const size_t bytes = (size_t)N * D * sizeof(float);

    std::printf("softmax: N=%u D=%u\n", N, D);

    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    vku::Buffer X = vku::make_host_ssbo(ctx, bytes);
    vku::Buffer Y = vku::make_host_ssbo(ctx, bytes);
    std::vector<float> x_host((size_t)N * D);
    read_bin(x_path, x_host.data(), bytes);
    vku::upload(ctx, X, x_host.data(), bytes);

    VkDescriptorSetLayoutBinding bindings[2]{};
    for (int i = 0; i < 2; ++i) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 2; dsli.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &dsl));

    struct PC { uint32_t D; } pc = { D };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(PC);

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    auto spv = vku::load_spirv("shaders/softmax_f32.comp.spv");
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));

    VkDescriptorBufferInfo dbi[2] = {
        {X.buf, 0, VK_WHOLE_SIZE},
        {Y.buf, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[2]{};
    for (int i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 2, writes, 0, nullptr);

    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, N, 1, 1);
    });
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    std::vector<float> y_host((size_t)N * D);
    vku::download(ctx, Y, y_host.data(), bytes);
    write_bin(y_path, y_host.data(), bytes);

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, X);
    vku::destroy_buffer(ctx, Y);
    vku::destroy_context(ctx);
    return 0;
}
