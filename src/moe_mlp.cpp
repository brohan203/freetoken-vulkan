// moe_mlp.cpp — CLI wrapper for the fused MoE MLP kernel.
//
// Usage: moe_mlp <x.bin> <indices.bin> <weights.bin>
//                <W_gate.bin> <W_up.bin> <W_down.bin>
//                <y.bin> <T> <D> <Dff> <E> <K>

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
    if (argc != 13) {
        std::fprintf(stderr,
            "usage: %s <x.bin> <indices.bin> <weights.bin> "
            "<W_gate.bin> <W_up.bin> <W_down.bin> "
            "<y.bin> <T> <D> <Dff> <E> <K>\n", argv[0]);
        return 2;
    }
    const uint32_t T   = (uint32_t)std::strtoul(argv[8],  nullptr, 10);
    const uint32_t D   = (uint32_t)std::strtoul(argv[9],  nullptr, 10);
    const uint32_t Dff = (uint32_t)std::strtoul(argv[10], nullptr, 10);
    const uint32_t E   = (uint32_t)std::strtoul(argv[11], nullptr, 10);
    const uint32_t K   = (uint32_t)std::strtoul(argv[12], nullptr, 10);

    const size_t x_bytes       = (size_t)T * D * sizeof(float);
    const size_t indices_bytes = (size_t)T * K * sizeof(uint32_t);
    const size_t weights_bytes = (size_t)T * K * sizeof(float);
    const size_t wgu_bytes     = (size_t)E * Dff * D * sizeof(float);
    const size_t wd_bytes      = (size_t)E * D * Dff * sizeof(float);
    const size_t y_bytes       = x_bytes;

    std::printf("moe_mlp: T=%u D=%u Dff=%u E=%u K=%u\n", T, D, Dff, E, K);
    std::printf("  W_gate size: %.2f MB per tensor\n",
                wgu_bytes / (1024.0 * 1024.0));

    vku::Context ctx = vku::create_context();
    std::printf("device: %s\n", ctx.properties.deviceName);

    vku::Buffer X  = vku::make_host_ssbo(ctx, x_bytes);
    vku::Buffer I  = vku::make_host_ssbo(ctx, indices_bytes);
    vku::Buffer W  = vku::make_host_ssbo(ctx, weights_bytes);
    vku::Buffer WG = vku::make_host_ssbo(ctx, wgu_bytes);
    vku::Buffer WU = vku::make_host_ssbo(ctx, wgu_bytes);
    vku::Buffer WD = vku::make_host_ssbo(ctx, wd_bytes);
    vku::Buffer Y  = vku::make_host_ssbo(ctx, y_bytes);

    std::vector<float>    x_host((size_t)T * D);
    std::vector<uint32_t> i_host((size_t)T * K);
    std::vector<float>    w_host((size_t)T * K);
    std::vector<float>    wg_host((size_t)E * Dff * D);
    std::vector<float>    wu_host((size_t)E * Dff * D);
    std::vector<float>    wd_host((size_t)E * D * Dff);
    read_bin(argv[1], x_host.data(),  x_bytes);
    read_bin(argv[2], i_host.data(),  indices_bytes);
    read_bin(argv[3], w_host.data(),  weights_bytes);
    read_bin(argv[4], wg_host.data(), wgu_bytes);
    read_bin(argv[5], wu_host.data(), wgu_bytes);
    read_bin(argv[6], wd_host.data(), wd_bytes);
    vku::upload(ctx, X,  x_host.data(),  x_bytes);
    vku::upload(ctx, I,  i_host.data(),  indices_bytes);
    vku::upload(ctx, W,  w_host.data(),  weights_bytes);
    vku::upload(ctx, WG, wg_host.data(), wgu_bytes);
    vku::upload(ctx, WU, wu_host.data(), wgu_bytes);
    vku::upload(ctx, WD, wd_host.data(), wd_bytes);

    VkDescriptorSetLayoutBinding bindings[7]{};
    for (int i = 0; i < 7; ++i) {
        bindings[i].binding = (uint32_t)i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 7; dsli.pBindings = bindings;
    VkDescriptorSetLayout dsl;
    VKU_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dsli, nullptr, &dsl));

    struct PC { uint32_t T, D, Dff, E, K; } pc = { T, D, Dff, E, K };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.size = sizeof(PC);
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VKU_CHECK(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipeline_layout));

    auto spv = vku::load_spirv("shaders/moe_mlp_f32.comp.spv");
    VkShaderModule sm = vku::make_shader_module(ctx.device, spv);
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VKU_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VKU_CHECK(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VKU_CHECK(vkAllocateDescriptorSets(ctx.device, &dai, &dset));

    VkDescriptorBufferInfo dbi[7] = {
        {X.buf,  0, VK_WHOLE_SIZE}, {I.buf,  0, VK_WHOLE_SIZE},
        {W.buf,  0, VK_WHOLE_SIZE}, {WG.buf, 0, VK_WHOLE_SIZE},
        {WU.buf, 0, VK_WHOLE_SIZE}, {WD.buf, 0, VK_WHOLE_SIZE},
        {Y.buf,  0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[7]{};
    for (int i = 0; i < 7; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = dset; writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 7, writes, 0, nullptr);

    // Warmup + timed dispatch
    (void)vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, T, 1, 1);
    });
    double ms = vku::submit_and_wait(ctx, [&](VkCommandBuffer cbuf) {
        vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PC), &pc);
        vkCmdDispatch(cbuf, T, 1, 1);
    });
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    std::vector<float> y_host((size_t)T * D);
    vku::download(ctx, Y, y_host.data(), y_bytes);
    write_bin(argv[7], y_host.data(), y_bytes);

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, sm, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, dsl, nullptr);
    vku::destroy_buffer(ctx, X);
    vku::destroy_buffer(ctx, I);
    vku::destroy_buffer(ctx, W);
    vku::destroy_buffer(ctx, WG);
    vku::destroy_buffer(ctx, WU);
    vku::destroy_buffer(ctx, WD);
    vku::destroy_buffer(ctx, Y);
    vku::destroy_context(ctx);
    return 0;
}
