//! Minimal Vulkan 1.3 context for the TLAS-isolation benchmark.
//!
//! The C API is bound by translate-c output imported as the `vk` module. The
//! loader is linked directly (`libvulkan.so.1`), so the core entry points are
//! ordinary extern symbols; only the KHR acceleration-structure entry points are
//! resolved through `vkGetDeviceProcAddr`.
//!
//! Nothing here is shared with svulkan2.

const std = @import("std");
const vk = @import("vk");

/// Unwraps a `PFN_*` typedef, which translate-c makes optional.
fn NonOpt(comptime Pfn: type) type {
    return @typeInfo(Pfn).optional.child;
}

/// Resolves a device-level entry point, which is exposed only as an extension in
/// the header. Core 1.2/1.3 entry points are ordinary extern symbols instead.
fn loadFn(comptime Pfn: type, device: vk.VkDevice, comptime name: [:0]const u8) !NonOpt(Pfn) {
    const raw: Pfn = @ptrCast(vk.vkGetDeviceProcAddr(device, name));
    return raw orelse error.MissingFunctionPointer;
}

pub const Buffer = struct {
    buffer: vk.VkBuffer,
    mem: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    /// Host pointer, only for host-visible allocations.
    mapped: ?[*]u8,
    address: vk.VkDeviceAddress,
};

pub const AccelerationStructure = struct {
    handle: vk.VkAccelerationStructureKHR,
    buffer: vk.VkBuffer,
    mem: vk.VkDeviceMemory,
    size: vk.VkDeviceSize,
    address: vk.VkDeviceAddress,
};

pub const Context = struct {
    instance: vk.VkInstance,
    physical: vk.VkPhysicalDevice,
    device: vk.VkDevice,
    queue: vk.VkQueue,
    queue_family: u32,
    cmd_pool: vk.VkCommandPool,
    cmd: vk.VkCommandBuffer,
    /// Reused for every submission so the measured submit path allocates nothing.
    fence: vk.VkFence,
    properties: vk.VkPhysicalDeviceProperties,
    memory_properties: vk.VkPhysicalDeviceMemoryProperties,
    as_properties: vk.VkPhysicalDeviceAccelerationStructurePropertiesKHR,
    maintenance3: vk.VkPhysicalDeviceMaintenance3Properties,
    validation_enabled: bool,

    vkCreateAccelerationStructureKHR: NonOpt(vk.PFN_vkCreateAccelerationStructureKHR),
    vkDestroyAccelerationStructureKHR: NonOpt(vk.PFN_vkDestroyAccelerationStructureKHR),
    vkGetAccelerationStructureBuildSizesKHR: NonOpt(vk.PFN_vkGetAccelerationStructureBuildSizesKHR),
    vkCmdBuildAccelerationStructuresKHR: NonOpt(vk.PFN_vkCmdBuildAccelerationStructuresKHR),
    vkGetAccelerationStructureDeviceAddressKHR: NonOpt(vk.PFN_vkGetAccelerationStructureDeviceAddressKHR),

    pub fn init(with_validation: bool) !Context {
        var self: Context = undefined;
        self.validation_enabled = false;

        var app_info = std.mem.zeroes(vk.VkApplicationInfo);
        app_info.sType = vk.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "tlas-isolation-bench";
        app_info.applicationVersion = 1;
        app_info.pEngineName = "none";
        app_info.engineVersion = 1;
        app_info.apiVersion = vk.VK_API_VERSION_1_3;

        var layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        var layer_count: u32 = 0;
        if (with_validation) {
            if (try layerAvailable("VK_LAYER_KHRONOS_validation")) {
                layer_count = 1;
                self.validation_enabled = true;
            } else {
                std.debug.print("note: VK_LAYER_KHRONOS_validation not installed, running without validation\n", .{});
            }
        }

        var inst_exts = [_][*:0]const u8{vk.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
        var ci = std.mem.zeroes(vk.VkInstanceCreateInfo);
        ci.sType = vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &app_info;
        ci.enabledLayerCount = layer_count;
        if (layer_count != 0) ci.ppEnabledLayerNames = &layers;
        ci.enabledExtensionCount = inst_exts.len;
        ci.ppEnabledExtensionNames = &inst_exts;

        var instance: vk.VkInstance = null;
        try check(vk.vkCreateInstance(&ci, null, &instance), "vkCreateInstance");
        self.instance = instance;

        self.physical = try pickGpu(instance);

        vk.vkGetPhysicalDeviceProperties(self.physical, &self.properties);
        vk.vkGetPhysicalDeviceMemoryProperties(self.physical, &self.memory_properties);

        var as_props = std.mem.zeroes(vk.VkPhysicalDeviceAccelerationStructurePropertiesKHR);
        as_props.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        var m3 = std.mem.zeroes(vk.VkPhysicalDeviceMaintenance3Properties);
        m3.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
        var props2 = std.mem.zeroes(vk.VkPhysicalDeviceProperties2);
        props2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &as_props;
        as_props.pNext = &m3;
        vk.vkGetPhysicalDeviceProperties2(self.physical, &props2);
        self.as_properties = as_props;
        self.maintenance3 = m3;

        var as_feat = std.mem.zeroes(vk.VkPhysicalDeviceAccelerationStructureFeaturesKHR);
        as_feat.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        var rq_feat = std.mem.zeroes(vk.VkPhysicalDeviceRayQueryFeaturesKHR);
        rq_feat.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        var bda_feat = std.mem.zeroes(vk.VkPhysicalDeviceBufferDeviceAddressFeatures);
        bda_feat.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        var di_feat = std.mem.zeroes(vk.VkPhysicalDeviceDescriptorIndexingFeatures);
        di_feat.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        var feats2 = std.mem.zeroes(vk.VkPhysicalDeviceFeatures2);
        feats2.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feats2.pNext = &as_feat;
        as_feat.pNext = &rq_feat;
        rq_feat.pNext = &bda_feat;
        bda_feat.pNext = &di_feat;
        vk.vkGetPhysicalDeviceFeatures2(self.physical, &feats2);

        if (as_feat.accelerationStructure == 0) return error.NoAccelerationStructure;
        if (rq_feat.rayQuery == 0) return error.NoRayQuery;
        if (bda_feat.bufferDeviceAddress == 0) return error.NoBufferDeviceAddress;
        if (di_feat.runtimeDescriptorArray == 0) return error.NoRuntimeDescriptorArray;
        if (di_feat.descriptorBindingPartiallyBound == 0) return error.NoPartiallyBound;
        if (di_feat.descriptorBindingVariableDescriptorCount == 0) return error.NoVariableDescriptorCount;

        var family_count: u32 = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical, &family_count, null);
        var families: [64]vk.VkQueueFamilyProperties = undefined;
        var fam_actual: u32 = @min(family_count, families.len);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(self.physical, &fam_actual, &families);
        var family: ?u32 = null;
        for (families[0..fam_actual], 0..) |f, i| {
            if (f.queueFlags & vk.VK_QUEUE_COMPUTE_BIT != 0) {
                family = @intCast(i);
                break;
            }
        }
        self.queue_family = family orelse return error.NoComputeQueue;

        const device_exts = [_][*:0]const u8{
            vk.VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            vk.VK_KHR_RAY_QUERY_EXTENSION_NAME,
            vk.VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        };

        // Enable exactly what the shader and the AS builds need.
        as_feat.accelerationStructure = vk.VK_TRUE;
        as_feat.descriptorBindingAccelerationStructureUpdateAfterBind = vk.VK_FALSE;
        rq_feat.rayQuery = vk.VK_TRUE;
        bda_feat.bufferDeviceAddress = vk.VK_TRUE;
        di_feat.runtimeDescriptorArray = vk.VK_TRUE;
        di_feat.descriptorBindingPartiallyBound = vk.VK_TRUE;
        di_feat.descriptorBindingVariableDescriptorCount = vk.VK_TRUE;
        di_feat.shaderStorageBufferArrayNonUniformIndexing = vk.VK_TRUE;
        feats2.pNext = &as_feat;
        feats2.features = std.mem.zeroes(vk.VkPhysicalDeviceFeatures);

        const prio = [1]f32{1.0};
        var qci = std.mem.zeroes(vk.VkDeviceQueueCreateInfo);
        qci.sType = vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = self.queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        var dci = std.mem.zeroes(vk.VkDeviceCreateInfo);
        dci.sType = vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &feats2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = device_exts.len;
        dci.ppEnabledExtensionNames = &device_exts;
        dci.pEnabledFeatures = null;

        var device: vk.VkDevice = null;
        try check(vk.vkCreateDevice(self.physical, &dci, null, &device), "vkCreateDevice");
        self.device = device;
        vk.vkGetDeviceQueue(device, self.queue_family, 0, &self.queue);

        self.vkCreateAccelerationStructureKHR = try loadFn(vk.PFN_vkCreateAccelerationStructureKHR, device, "vkCreateAccelerationStructureKHR");
        self.vkDestroyAccelerationStructureKHR = try loadFn(vk.PFN_vkDestroyAccelerationStructureKHR, device, "vkDestroyAccelerationStructureKHR");
        self.vkGetAccelerationStructureBuildSizesKHR = try loadFn(vk.PFN_vkGetAccelerationStructureBuildSizesKHR, device, "vkGetAccelerationStructureBuildSizesKHR");
        self.vkCmdBuildAccelerationStructuresKHR = try loadFn(vk.PFN_vkCmdBuildAccelerationStructuresKHR, device, "vkCmdBuildAccelerationStructuresKHR");
        self.vkGetAccelerationStructureDeviceAddressKHR = try loadFn(vk.PFN_vkGetAccelerationStructureDeviceAddressKHR, device, "vkGetAccelerationStructureDeviceAddressKHR");

        var cpci = std.mem.zeroes(vk.VkCommandPoolCreateInfo);
        cpci.sType = vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = self.queue_family;
        try check(vk.vkCreateCommandPool(device, &cpci, null, &self.cmd_pool), "vkCreateCommandPool");

        var cbai = std.mem.zeroes(vk.VkCommandBufferAllocateInfo);
        cbai.sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = self.cmd_pool;
        cbai.level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        try check(vk.vkAllocateCommandBuffers(device, &cbai, &self.cmd), "vkAllocateCommandBuffers");

        var fci = std.mem.zeroes(vk.VkFenceCreateInfo);
        fci.sType = vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        try check(vk.vkCreateFence(device, &fci, null, &self.fence), "vkCreateFence");

        return self;
    }

    pub fn deinit(self: *Context) void {
        vk.vkDestroyFence(self.device, self.fence, null);
        vk.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        vk.vkDestroyDevice(self.device, null);
        vk.vkDestroyInstance(self.instance, null);
    }

    /// Total size of device-local heaps, the budget the AS allocations compete for.
    pub fn deviceLocalHeapBytes(self: *const Context) u64 {
        var total: u64 = 0;
        const mp = self.memory_properties;
        for (0..@min(mp.memoryHeapCount, 16)) |i| {
            if (mp.memoryHeaps[i].flags & vk.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                total += mp.memoryHeaps[i].size;
            }
        }
        return total;
    }

    pub fn createBuffer(
        self: *Context,
        size: vk.VkDeviceSize,
        usage: vk.VkBufferUsageFlags,
        required_flags: vk.VkMemoryPropertyFlags,
        map: bool,
    ) !Buffer {
        var bci = std.mem.zeroes(vk.VkBufferCreateInfo);
        bci.sType = vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE;
        var buffer: vk.VkBuffer = null;
        try check(vk.vkCreateBuffer(self.device, &bci, null, &buffer), "vkCreateBuffer");
        errdefer vk.vkDestroyBuffer(self.device, buffer, null);

        var req: vk.VkMemoryRequirements = undefined;
        vk.vkGetBufferMemoryRequirements(self.device, buffer, &req);

        // Device addresses need the allocate-time device-address flag.
        const needs_bda = usage & vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT != 0;
        var flags_info = std.mem.zeroes(vk.VkMemoryAllocateFlagsInfo);
        flags_info.sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flags_info.flags = vk.VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        var mai = std.mem.zeroes(vk.VkMemoryAllocateInfo);
        mai.sType = vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        if (needs_bda) mai.pNext = &flags_info;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = try findMemoryType(self.memory_properties, req.memoryTypeBits, required_flags);
        var mem: vk.VkDeviceMemory = null;
        try check(vk.vkAllocateMemory(self.device, &mai, null, &mem), "vkAllocateMemory");
        errdefer vk.vkFreeMemory(self.device, mem, null);
        try check(vk.vkBindBufferMemory(self.device, buffer, mem, 0), "vkBindBufferMemory");

        var mapped: ?[*]u8 = null;
        if (map) {
            var ptr: ?*anyopaque = null;
            try check(vk.vkMapMemory(self.device, mem, 0, vk.VK_WHOLE_SIZE, 0, &ptr), "vkMapMemory");
            mapped = @ptrCast(ptr);
        }

        var address: vk.VkDeviceAddress = 0;
        if (needs_bda) {
            var info = std.mem.zeroes(vk.VkBufferDeviceAddressInfo);
            info.sType = vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            info.buffer = buffer;
            address = vk.vkGetBufferDeviceAddress(self.device, &info);
        }
        return .{ .buffer = buffer, .mem = mem, .size = size, .mapped = mapped, .address = address };
    }

    pub fn destroyBuffer(self: *Context, b: Buffer) void {
        if (b.mapped != null) vk.vkUnmapMemory(self.device, b.mem);
        vk.vkDestroyBuffer(self.device, b.buffer, null);
        vk.vkFreeMemory(self.device, b.mem, null);
    }

    /// Creates storage for an acceleration structure and returns it with its
    /// device address.
    pub fn createAccelerationStructure(
        self: *Context,
        kind: vk.VkAccelerationStructureTypeKHR,
        size: vk.VkDeviceSize,
    ) !AccelerationStructure {
        const storage = try self.createBuffer(
            size,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            false,
        );
        errdefer self.destroyBuffer(storage);

        var info = std.mem.zeroes(vk.VkAccelerationStructureCreateInfoKHR);
        info.sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        info.buffer = storage.buffer;
        info.offset = 0;
        info.size = size;
        info.type = kind;
        var handle: vk.VkAccelerationStructureKHR = null;
        try check(self.vkCreateAccelerationStructureKHR(self.device, &info, null, &handle), "vkCreateAccelerationStructureKHR");
        errdefer self.vkDestroyAccelerationStructureKHR(self.device, handle, null);

        var addr_info = std.mem.zeroes(vk.VkAccelerationStructureDeviceAddressInfoKHR);
        addr_info.sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addr_info.accelerationStructure = handle;
        const addr = self.vkGetAccelerationStructureDeviceAddressKHR(self.device, &addr_info);
        return .{ .handle = handle, .buffer = storage.buffer, .mem = storage.mem, .size = size, .address = addr };
    }

    pub fn destroyAccelerationStructure(self: *Context, a: AccelerationStructure) void {
        if (a.handle != null) self.vkDestroyAccelerationStructureKHR(self.device, a.handle, null);
        vk.vkDestroyBuffer(self.device, a.buffer, null);
        vk.vkFreeMemory(self.device, a.mem, null);
    }

    /// Runs a recorded command buffer to completion on the compute queue.
    pub fn submitAndWait(self: *Context) !void {
        var si = std.mem.zeroes(vk.VkSubmitInfo);
        si.sType = vk.VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &self.cmd;
        // The fence must be unsignaled when submitted; it stays signaled after the
        // previous wait.
        try check(vk.vkResetFences(self.device, 1, &self.fence), "vkResetFences");
        try check(vk.vkQueueSubmit(self.queue, 1, &si, self.fence), "vkQueueSubmit");
        try check(vk.vkWaitForFences(self.device, 1, &self.fence, vk.VK_TRUE, std.math.maxInt(u64)), "vkWaitForFences");
    }

    pub fn beginCommands(self: *Context) !void {
        try check(vk.vkResetCommandBuffer(self.cmd, 0), "vkResetCommandBuffer");
        var begin = std.mem.zeroes(vk.VkCommandBufferBeginInfo);
        begin.sType = vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        try check(vk.vkBeginCommandBuffer(self.cmd, &begin), "vkBeginCommandBuffer");
    }

    pub fn endCommands(self: *Context) !void {
        try check(vk.vkEndCommandBuffer(self.cmd), "vkEndCommandBuffer");
    }

    /// Full barrier between the AS build and the shader reads.
    pub fn asBuildToComputeBarrier(self: *Context) void {
        var barrier = std.mem.zeroes(vk.VkMemoryBarrier);
        barrier.sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | vk.VK_ACCESS_SHADER_READ_BIT;
        vk.vkCmdPipelineBarrier(
            self.cmd,
            vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }
};

pub fn check(result: vk.VkResult, comptime what: []const u8) !void {
    if (result != vk.VK_SUCCESS) {
        std.debug.print("Vulkan call failed: {s} -> VkResult {d}\n", .{ what, result });
        return error.VulkanFailure;
    }
}

pub fn findMemoryType(
    mem_props: vk.VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk.VkMemoryPropertyFlags,
) !u32 {
    var i: u32 = 0;
    while (i < @min(mem_props.memoryTypeCount, 32)) : (i += 1) {
        if (type_bits & (@as(u32, 1) << @intCast(i)) == 0) continue;
        if (mem_props.memoryTypes[i].propertyFlags & required == required) return i;
    }
    return error.NoSuitableMemoryType;
}

fn layerAvailable(name: [:0]const u8) !bool {
    var count: u32 = 0;
    if (vk.vkEnumerateInstanceLayerProperties(&count, null) != vk.VK_SUCCESS) return false;
    var props: [64]vk.VkLayerProperties = undefined;
    var actual: u32 = @min(count, props.len);
    if (vk.vkEnumerateInstanceLayerProperties(&actual, &props) != vk.VK_SUCCESS) return false;
    for (props[0..actual]) |p| {
        if (std.mem.eql(u8, std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&p.layerName)), 0), name)) return true;
    }
    return false;
}

fn pickGpu(instance: vk.VkInstance) !vk.VkPhysicalDevice {
    var count: u32 = 0;
    try check(vk.vkEnumeratePhysicalDevices(instance, &count, null), "vkEnumeratePhysicalDevices");
    if (count == 0) return error.NoVulkanDevice;
    var gpus: [32]vk.VkPhysicalDevice = undefined;
    var actual: u32 = @min(count, gpus.len);
    try check(vk.vkEnumeratePhysicalDevices(instance, &actual, &gpus), "vkEnumeratePhysicalDevices");

    // Prefer a discrete NVIDIA GPU.
    var best: ?vk.VkPhysicalDevice = null;
    var best_score: u32 = 0;
    for (gpus[0..actual]) |gpu| {
        var props: vk.VkPhysicalDeviceProperties = undefined;
        vk.vkGetPhysicalDeviceProperties(gpu, &props);
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&props.deviceName)), 0);
        if (props.deviceType != vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        const score: u32 = if (std.mem.indexOf(u8, name, "NVIDIA") != null) 2 else 1;
        if (score > best_score) {
            best_score = score;
            best = gpu;
        }
    }
    const chosen = best orelse return error.NoDiscreteGpu;

    var props: vk.VkPhysicalDeviceProperties = undefined;
    vk.vkGetPhysicalDeviceProperties(chosen, &props);
    const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&props.deviceName)), 0);
    std.debug.print("gpu: {s} driver {d}.{d}.{d} api {d}.{d}.{d}\n", .{
        name,
        vk.VK_API_VERSION_MAJOR(props.driverVersion),
        vk.VK_API_VERSION_MINOR(props.driverVersion),
        vk.VK_API_VERSION_PATCH(props.driverVersion),
        vk.VK_API_VERSION_MAJOR(props.apiVersion),
        vk.VK_API_VERSION_MINOR(props.apiVersion),
        vk.VK_API_VERSION_PATCH(props.apiVersion),
    });
    return chosen;
}
