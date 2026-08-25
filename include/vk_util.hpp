#pragma once
// vk_util.hpp - tiny header-only Vulkan compute utility layer.
//
// vector_add.cpp shows every step raw. Every subsequent kernel would drown
// in the same 200 lines of boilerplate, so we lift instance/device/queue/
// buffer/shader creation into this header. Kernel .cpp files then focus on
// the kernel-specific bits: descriptor layout, push constants, dispatch.
//
// Deliberately minimal - no exceptions, no RAII wrappers, no CRTP. Read it
// top-to-bottom and every line maps to a Vulkan call you already saw in
// vector_add.cpp.

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vku {

#define VKU_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "Vulkan error %d at %s:%d\n", _r, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// ============================================================
// Context - everything you need once, kept around for the whole run.
// ============================================================
struct Context {
    VkInstance                 instance         = VK_NULL_HANDLE;
    VkPhysicalDevice           physical_device  = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties       = {};
    VkDevice                   device           = VK_NULL_HANDLE;
    VkQueue                    queue            = VK_NULL_HANDLE;
    uint32_t                   queue_family     = UINT32_MAX;
    VkCommandPool              command_pool     = VK_NULL_HANDLE;
};

inline Context create_context() {
    Context ctx{};

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "freetoken-vulkan";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VKU_CHECK(vkCreateInstance(&ii, nullptr, &ctx.instance));

    // Pick a physical device - prefer discrete.
    uint32_t n = 0;
    VKU_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &n, nullptr));
    if (n == 0) { std::fprintf(stderr, "no Vulkan GPU\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> pds(n);
    VKU_CHECK(vkEnumeratePhysicalDevices(ctx.instance, &n, pds.data()));
    ctx.physical_device = pds[0];
    for (auto& cand : pds) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(cand, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            ctx.physical_device = cand; break;
        }
    }
    vkGetPhysicalDeviceProperties(ctx.physical_device, &ctx.properties);

    // Find compute queue family.
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(ctx.physical_device, &qn, qfs.data());
    for (uint32_t i = 0; i < qn; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { ctx.queue_family = i; break; }
    }
    if (ctx.queue_family == UINT32_MAX) { std::fprintf(stderr, "no compute queue\n"); std::exit(1); }

    // Enable FP16 + INT8 features. Chain: Vulkan11Features (16-bit storage) ->
    // Vulkan12Features (shaderFloat16 + shaderInt8 + 8-bit storage) -> attached
    // to DeviceCreateInfo.pNext. 6800 XT / RDNA2 supports all four.
    // Int8 + 8-bit storage are needed for MXFP4/NVFP4 packed weights (uint8
    // blocks + uint8 scales).
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.shaderFloat16              = VK_TRUE;
    v12.shaderInt8                 = VK_TRUE;
    v12.storageBuffer8BitAccess    = VK_TRUE;
    v12.uniformAndStorageBuffer8BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.storageBuffer16BitAccess = VK_TRUE;
    v11.pNext = &v12;

    // Logical device + queue.
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = ctx.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qprio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pNext = &v11;   // enable the fp16 + 16bit-storage features
    VKU_CHECK(vkCreateDevice(ctx.physical_device, &dci, nullptr, &ctx.device));
    vkGetDeviceQueue(ctx.device, ctx.queue_family, 0, &ctx.queue);

    // Command pool.
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = ctx.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VKU_CHECK(vkCreateCommandPool(ctx.device, &cpci, nullptr, &ctx.command_pool));

    return ctx;
}

inline void destroy_context(Context& ctx) {
    if (ctx.command_pool) vkDestroyCommandPool(ctx.device, ctx.command_pool, nullptr);
    if (ctx.device)       vkDestroyDevice(ctx.device, nullptr);
    if (ctx.instance)     vkDestroyInstance(ctx.instance, nullptr);
    ctx = {};
}

// ============================================================
// Buffer helpers.
// ============================================================
struct Buffer {
    VkBuffer       buf  = VK_NULL_HANDLE;
    VkDeviceMemory mem  = VK_NULL_HANDLE;
    VkDeviceSize   size = 0;
};

inline uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t type_bits,
                                 VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    std::fprintf(stderr, "no matching memory type\n"); std::exit(1);
}

inline Buffer make_buffer(const Context& ctx, VkDeviceSize size,
                          VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
    Buffer b{};
    b.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKU_CHECK(vkCreateBuffer(ctx.device, &bi, nullptr, &b.buf));
    b.size = size;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(ctx.device, b.buf, &mr);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = find_memory_type(ctx.physical_device, mr.memoryTypeBits, props);
    VKU_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &b.mem));
    VKU_CHECK(vkBindBufferMemory(ctx.device, b.buf, b.mem, 0));
    return b;
}

inline void destroy_buffer(const Context& ctx, Buffer& b) {
    if (b.buf) vkDestroyBuffer(ctx.device, b.buf, nullptr);
    if (b.mem) vkFreeMemory(ctx.device, b.mem, nullptr);
    b = {};
}

// Host-visible+coherent shortcut - good enough for correctness testing.
// Real perf work will use device-local + staging buffers.
inline Buffer make_host_ssbo(const Context& ctx, VkDeviceSize size) {
    return make_buffer(ctx, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

// Device-local buffer, resides in VRAM. Faster for GPU compute but not
// directly writable from CPU; must be filled via a staging buffer +
// vkCmdCopyBuffer. Use for weights that live across many kernel calls.
inline Buffer make_device_ssbo(const Context& ctx, VkDeviceSize size) {
    return make_buffer(ctx, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// ============================================================
// Buffer pool: recycles HOST_VISIBLE VkBuffers across kernel calls.
//
// Problem: allocating and freeing hundreds of small VkBuffers per forward
// pass exhausts the driver's memory-allocation tracking on AMD/Windows,
// hitting VK_ERROR_OUT_OF_DEVICE_MEMORY around forward pass #3 when a
// large baseline (like resident MoE weights, ~10 GB) is already pinned.
//
// Solution: bucket buffers by log2 size. Acquire returns an existing free
// buffer of >= the requested size, or allocates a new one at the bucket
// size. Release puts it back on the free list for the same bucket instead
// of freeing.
//
// - Growth-only (never returns memory to the OS while the pool is alive)
// - Bucket boundaries are powers of 2 from 4 KiB up to 512 MiB
// - Oversized requests receive an exact-size unpooled allocation
// - Caps the free list per bucket to avoid unbounded growth of small buffers
// ============================================================
struct BufferPool {
    Context* ctx = nullptr;
    static constexpr uint32_t kNumBuckets = 18; // 4 KiB .. 512 MiB
    static constexpr size_t   kMaxPerBucket = 64;

    std::vector<Buffer> host_pool[kNumBuckets];

    static uint32_t bucket_of(size_t bytes) {
        // Smallest bucket = 4 KiB. Bucket i holds sizes (2^(11+i), 2^(12+i)].
        // Bucket 0: (4KB, 8KB]; 1: (8, 16]; ...; 17: (512MB, 1GB].
        // For bytes <= 4KB use bucket 0 (waste at most 4KB per allocation).
        uint32_t idx = 0;
        size_t cap = 4 * 1024;
        while (idx + 1 < kNumBuckets && cap < bytes) {
            cap *= 2;
            idx++;
        }
        return idx;
    }

    static size_t bucket_size(uint32_t idx) {
        size_t cap = 4 * 1024;
        for (uint32_t i = 0; i < idx; ++i) cap *= 2;
        return cap;
    }

    static size_t max_bucket_size() {
        return bucket_size(kNumBuckets - 1);
    }

    Buffer acquire_host(size_t bytes) {
        if (bytes > max_bucket_size()) {
            return make_host_ssbo(*ctx, bytes);
        }
        uint32_t idx = bucket_of(bytes);
        auto& bucket = host_pool[idx];
        if (!bucket.empty()) {
            Buffer b = bucket.back();
            bucket.pop_back();
            return b;
        }
        return make_host_ssbo(*ctx, bucket_size(idx));
    }

    void release_host(Buffer& b) {
        if (!b.buf) return;
        if ((size_t)b.size > max_bucket_size()) {
            destroy_buffer(*ctx, b);
            return;
        }
        uint32_t idx = bucket_of((size_t)b.size);
        auto& bucket = host_pool[idx];
        if (bucket.size() < kMaxPerBucket) {
            bucket.push_back(b);
            b = {};
        } else {
            destroy_buffer(*ctx, b);
        }
    }

    void destroy_all() {
        for (auto& bucket : host_pool) {
            for (auto& b : bucket) destroy_buffer(*ctx, b);
            bucket.clear();
        }
    }

    size_t total_bytes_held() const {
        size_t total = 0;
        for (const auto& bucket : host_pool) {
            for (const auto& b : bucket) total += (size_t)b.size;
        }
        return total;
    }
};

inline void upload(const Context& ctx, Buffer& b, const void* src, VkDeviceSize bytes) {
    if (bytes > b.size) {
        throw std::runtime_error("upload exceeds Vulkan buffer capacity");
    }
    void* p;
    VKU_CHECK(vkMapMemory(ctx.device, b.mem, 0, bytes, 0, &p));
    std::memcpy(p, src, (size_t)bytes);
    vkUnmapMemory(ctx.device, b.mem);
}

inline void download(const Context& ctx, Buffer& b, void* dst, VkDeviceSize bytes) {
    if (bytes > b.size) {
        throw std::runtime_error("download exceeds Vulkan buffer capacity");
    }
    void* p;
    VKU_CHECK(vkMapMemory(ctx.device, b.mem, 0, bytes, 0, &p));
    std::memcpy(dst, p, (size_t)bytes);
    vkUnmapMemory(ctx.device, b.mem);
}

// Upload from CPU into a device-local Buffer via a temporary staging buffer.
// Slow (one queue submit per call) but only needed at load time for weights
// that then live across many kernel invocations.
inline void upload_via_staging(const Context& ctx, Buffer& device_buf,
                                const void* src, VkDeviceSize bytes) {
    // 1. Allocate temp staging buffer, memcpy CPU -> staging.
    Buffer staging = make_host_ssbo(ctx, bytes);
    upload(ctx, staging, src, bytes);

    // 2. Record a one-shot command buffer that copies staging -> device.
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = ctx.command_pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VKU_CHECK(vkAllocateCommandBuffers(ctx.device, &ai, &cmd));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKU_CHECK(vkBeginCommandBuffer(cmd, &bi));

    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(cmd, staging.buf, device_buf.buf, 1, &copy);

    VKU_CHECK(vkEndCommandBuffer(cmd));

    // 3. Submit and wait.
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKU_CHECK(vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE));
    VKU_CHECK(vkQueueWaitIdle(ctx.queue));

    // 4. Cleanup.
    vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
    destroy_buffer(ctx, staging);
}

// ============================================================
// Shader helpers.
// ============================================================
inline std::vector<uint32_t> load_spirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    size_t bytes = (size_t)f.tellg();
    std::vector<uint32_t> code(bytes / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), bytes);
    return code;
}

inline VkShaderModule make_shader_module(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spv.size() * 4;
    smi.pCode = spv.data();
    VkShaderModule sm;
    VKU_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    return sm;
}

// ============================================================
// One-shot command buffer helper - begin, record via callback, submit, wait.
// Returns wall-clock ms of the queue submit+wait.
// ============================================================
// Reusable resources for submit_and_wait - allocated once, reused across
// every kernel call. Skipping vkAllocateCommandBuffers / vkFreeCommandBuffers
// each call is worth ~50-200 us; using a fence instead of vkQueueWaitIdle
// avoids draining the whole queue.
namespace detail {
    inline VkCommandBuffer& shared_cbuf() {
        static VkCommandBuffer cbuf = VK_NULL_HANDLE;
        return cbuf;
    }
    inline VkFence& shared_fence() {
        static VkFence f = VK_NULL_HANDLE;
        return f;
    }
}

template <typename Fn>
inline double submit_and_wait(const Context& ctx, Fn record) {
    VkCommandBuffer& cbuf = detail::shared_cbuf();
    VkFence& fence = detail::shared_fence();

    if (cbuf == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = ctx.command_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VKU_CHECK(vkAllocateCommandBuffers(ctx.device, &cbai, &cbuf));
    } else {
        VKU_CHECK(vkResetCommandBuffer(cbuf, 0));
    }
    if (fence == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VKU_CHECK(vkCreateFence(ctx.device, &fci, nullptr, &fence));
    }

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKU_CHECK(vkBeginCommandBuffer(cbuf, &cbbi));
    record(cbuf);
    VKU_CHECK(vkEndCommandBuffer(cbuf));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbuf;

    auto t0 = std::chrono::steady_clock::now();
    VKU_CHECK(vkQueueSubmit(ctx.queue, 1, &si, fence));
    // Busy-poll the fence with vkGetFenceStatus instead of vkWaitForFences.
    // On Windows AMD, vkWaitForFences appears to sleep+wake on the OS
    // scheduler tick (~15.6 ms), turning even microsecond-fast GPU
    // dispatches into ~18 ms round-trips. Busy-polling avoids that at
    // the cost of a hot CPU thread - fine for our per-call overhead.
    while (true) {
        VkResult r = vkGetFenceStatus(ctx.device, fence);
        if (r == VK_SUCCESS) break;
        if (r != VK_NOT_READY) { VKU_CHECK(r); }
        // Optional: _mm_pause() to reduce CPU heat, but keep tight.
    }
    VKU_CHECK(vkResetFences(ctx.device, 1, &fence));
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================
// GPU timestamp helper - measures actual on-GPU compute time via
// VkQueryPool, separate from queue submission + wait overhead.
//
// Usage:
//     TimestampQuery ts = make_timestamp_query(ctx);
//     double wall_ms = submit_with_timestamps(ctx, ts, [&](VkCommandBuffer cbuf){
//         cmd_reset_and_write_start(cbuf, ts);
//         // ... vkCmdDispatch calls ...
//         cmd_write_end(cbuf, ts);
//     });
//     double gpu_ms = read_gpu_ms(ctx, ts);
//     destroy_timestamp_query(ctx, ts);
//
// GPU-only ms is derived from two hardware timestamps captured
// pre/post compute. Requires the device to support timestamp queries
// on the compute queue (all discrete GPUs do; TimestampComputeAndGraphics
// or timestampPeriod must be > 0).
// ============================================================
struct TimestampQuery {
    VkQueryPool pool = VK_NULL_HANDLE;
    float       timestamp_period_ns = 1.0f;   // ns per tick, from device props
};

inline TimestampQuery make_timestamp_query(const Context& ctx) {
    TimestampQuery ts{};
    ts.timestamp_period_ns = ctx.properties.limits.timestampPeriod;
    if (ts.timestamp_period_ns == 0.0f) {
        std::fprintf(stderr,
            "warning: this queue does not support timestamps; results will be 0\n");
    }
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;   // start + end
    VKU_CHECK(vkCreateQueryPool(ctx.device, &qpci, nullptr, &ts.pool));
    return ts;
}

inline void destroy_timestamp_query(const Context& ctx, TimestampQuery& ts) {
    if (ts.pool) vkDestroyQueryPool(ctx.device, ts.pool, nullptr);
    ts = {};
}

// Reset queries + write timestamp at the top of the compute pipeline.
// Call at the START of your command buffer, before any dispatch.
inline void cmd_reset_and_write_start(VkCommandBuffer cbuf, const TimestampQuery& ts) {
    vkCmdResetQueryPool(cbuf, ts.pool, 0, 2);
    // TOP_OF_PIPE: measured when the command enters the pipeline. Fine for
    // compute-only workloads (no earlier graphics stage to matter).
    vkCmdWriteTimestamp(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts.pool, 0);
}

// Write end-timestamp after your dispatch calls. Uses BOTTOM_OF_PIPE so it
// fires only after the compute shader completes.
inline void cmd_write_end(VkCommandBuffer cbuf, const TimestampQuery& ts) {
    vkCmdWriteTimestamp(cbuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts.pool, 1);
}

// Same shape as submit_and_wait but the callback is expected to sandwich
// its dispatch calls between cmd_reset_and_write_start / cmd_write_end.
// Returns wall-clock ms (same as submit_and_wait). Use read_gpu_ms for
// the GPU-only figure.
template <typename Fn>
inline double submit_with_timestamps(const Context& ctx, const TimestampQuery& ts,
                                     Fn record) {
    (void)ts;   // unused here but part of the signature contract
    return submit_and_wait(ctx, record);
}

inline double read_gpu_ms(const Context& ctx, const TimestampQuery& ts) {
    uint64_t stamps[2] = {0, 0};
    VKU_CHECK(vkGetQueryPoolResults(
        ctx.device, ts.pool, 0, 2, sizeof(stamps), stamps,
        sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    const uint64_t delta_ticks = stamps[1] - stamps[0];
    const double   delta_ns    = (double)delta_ticks * ts.timestamp_period_ns;
    return delta_ns / 1.0e6;   // ns -> ms
}

} // namespace vku
