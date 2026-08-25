// vector_add.cpp — Vulkan host code for the vector_add compute shader.
//
// This is deliberately verbose. Every step is one of the Vulkan concepts
// you're here to learn: instance, physical device, logical device, queue,
// command buffer, descriptor set, pipeline, memory barriers. When you can
// read this file end-to-end and know why each step is here, you understand
// Vulkan compute.
//
// Build: see CMakeLists.txt at repo root. Run: vector_add.exe [N]
// (default N = 1<<20). Compares GPU result against CPU reference, prints
// max abs diff and pass/fail.

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <chrono>

// -------- tiny error-check macro --------
#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// -------- read a SPIR-V binary from disk --------
static std::vector<uint32_t> load_spirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    size_t bytes = (size_t)f.tellg();
    if (bytes % 4 != 0) { std::fprintf(stderr, "SPIR-V not 4-byte aligned\n"); std::exit(1); }
    std::vector<uint32_t> code(bytes / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), bytes);
    return code;
}

// -------- pick a memory type that matches requirements + property flags --------
static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t type_bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    std::fprintf(stderr, "no matching memory type\n"); std::exit(1);
}

// -------- allocate a buffer + backing memory --------
struct Buffer {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

static Buffer make_buffer(VkDevice dev, VkPhysicalDevice pd, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buffer b{};
    b.size = size;

    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.buf, &mr);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory_type(pd, mr.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    return b;
}

int main(int argc, char** argv) {
    const uint32_t N = (argc > 1) ? (uint32_t)std::atoi(argv[1]) : (1u << 20);
    std::printf("vector_add: N = %u elements (%.2f MB per buffer)\n",
                N, (N * sizeof(float)) / (1024.0 * 1024.0));

    // ============================================================
    // 1. Instance — the Vulkan loader's entry point.
    // ============================================================
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vector_add";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;

    VkInstance instance;
    VK_CHECK(vkCreateInstance(&ii, nullptr, &instance));

    // ============================================================
    // 2. Physical device — pick a GPU. Prefer discrete.
    // ============================================================
    uint32_t pd_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, nullptr));
    if (pd_count == 0) { std::fprintf(stderr, "no Vulkan GPU\n"); return 1; }
    std::vector<VkPhysicalDevice> pds(pd_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, pds.data()));

    VkPhysicalDevice pd = pds[0];
    for (auto& cand : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(cand, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = cand; break; }
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s (vendor 0x%04x, driver 0x%08x)\n",
                props.deviceName, props.vendorID, props.driverVersion);

    // ============================================================
    // 3. Queue family — find one that supports compute.
    // ============================================================
    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs.data());
    uint32_t qf_index = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf_index = i; break; }
    }
    if (qf_index == UINT32_MAX) { std::fprintf(stderr, "no compute queue\n"); return 1; }
    std::printf("compute queue family: %u\n", qf_index);

    // ============================================================
    // 4. Logical device + queue handle.
    // ============================================================
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qf_index;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qprio;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice dev;
    VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));

    VkQueue queue;
    vkGetDeviceQueue(dev, qf_index, 0, &queue);

    // ============================================================
    // 5. Allocate 3 host-visible+coherent buffers (a, b, c).
    //    Real ML kernels would use device-local + staging. This is
    //    the smallest possible working version.
    // ============================================================
    const VkDeviceSize bytes = N * sizeof(float);
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags host_visible =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    Buffer A = make_buffer(dev, pd, bytes, ssbo, host_visible);
    Buffer B = make_buffer(dev, pd, bytes, ssbo, host_visible);
    Buffer C = make_buffer(dev, pd, bytes, ssbo, host_visible);

    // Fill A and B on the CPU side.
    float* a_ptr; float* b_ptr;
    VK_CHECK(vkMapMemory(dev, A.mem, 0, bytes, 0, (void**)&a_ptr));
    VK_CHECK(vkMapMemory(dev, B.mem, 0, bytes, 0, (void**)&b_ptr));
    for (uint32_t i = 0; i < N; ++i) { a_ptr[i] = (float)i * 0.001f; b_ptr[i] = (float)(N - i) * 0.002f; }
    vkUnmapMemory(dev, A.mem);
    vkUnmapMemory(dev, B.mem);

    // ============================================================
    // 6. Descriptor set layout — describes the shader's binding slots.
    // ============================================================
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
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dsli, nullptr, &dsl));

    // ============================================================
    // 7. Pipeline layout — descriptor sets + push constants.
    // ============================================================
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(uint32_t); // matches `layout(push_constant) uniform Params { uint n; }`

    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeline_layout));

    // ============================================================
    // 8. Shader module + compute pipeline.
    // ============================================================
    auto spv = load_spirv("shaders/vector_add.comp.spv");
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spv.size() * 4;
    smi.pCode = spv.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = sm;
    cpi.stage.pName = "main";
    cpi.layout = pipeline_layout;
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline));

    // ============================================================
    // 9. Descriptor pool + descriptor set (bind our 3 buffers to the shader).
    // ============================================================
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));

    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &dset));

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
    vkUpdateDescriptorSets(dev, 3, writes, 0, nullptr);

    // ============================================================
    // 10. Command pool + buffer, record commands, submit, wait.
    // ============================================================
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qf_index;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool;
    VK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cbuf;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cbuf));

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cbuf, &cbbi));

    vkCmdBindPipeline(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cbuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                            0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cbuf, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(uint32_t), &N);

    // Dispatch ceil(N / 256) workgroups. Each workgroup has 256 invocations
    // (from local_size_x = 256 in the shader).
    const uint32_t WG = 256;
    const uint32_t groups = (N + WG - 1) / WG;
    vkCmdDispatch(cbuf, groups, 1, 1);

    VK_CHECK(vkEndCommandBuffer(cbuf));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbuf;

    auto t0 = std::chrono::steady_clock::now();
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("GPU dispatch + wait: %.3f ms\n", ms);

    // ============================================================
    // 11. Read back and verify against CPU reference.
    // ============================================================
    float* c_ptr;
    VK_CHECK(vkMapMemory(dev, C.mem, 0, bytes, 0, (void**)&c_ptr));

    // Re-map A and B to compute reference (we could keep them mapped earlier,
    // but this is the pattern you'd use with device-local staging too).
    VK_CHECK(vkMapMemory(dev, A.mem, 0, bytes, 0, (void**)&a_ptr));
    VK_CHECK(vkMapMemory(dev, B.mem, 0, bytes, 0, (void**)&b_ptr));

    // Reference in FP32 (matching GPU precision exactly). IEEE 754 FP32 add
    // is deterministic per-element, so we expect bit-exact agreement. Using a
    // double reference here would conflate GPU compute errors with the
    // ~1 ULP FP32 rounding, so we deliberately avoid that.
    double max_abs_diff = 0.0;
    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float ref = a_ptr[i] + b_ptr[i];
        double diff = std::fabs((double)c_ptr[i] - (double)ref);
        if (diff > 0.0) ++mismatch_count;
        if (diff > max_abs_diff) max_abs_diff = diff;
    }
    std::printf("mismatches: %u / %u  (max abs diff: %.3e)  ->  %s\n",
                mismatch_count, N, max_abs_diff,
                mismatch_count == 0 ? "PASS" : "FAIL");

    vkUnmapMemory(dev, A.mem);
    vkUnmapMemory(dev, B.mem);
    vkUnmapMemory(dev, C.mem);

    // ============================================================
    // 12. Cleanup (Vulkan is strict — leak these and validation layers scream).
    // ============================================================
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    for (Buffer* b : {&A, &B, &C}) {
        vkDestroyBuffer(dev, b->buf, nullptr);
        vkFreeMemory(dev, b->mem, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(instance, nullptr);
    return (mismatch_count == 0) ? 0 : 1;
}
