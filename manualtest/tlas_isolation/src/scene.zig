//! Scene layout, TLAS management and the per-frame measurement loop.
//!
//! Both isolation modes share everything except where instances are placed and
//! how many top-level acceleration structures exist:
//!
//!   A: ONE TLAS holding N*(M+2) instances; environment i translated by a
//!      centred-grid cell offset, and its rays clipped to that cell.
//!   B: N TLASes of M+2 instances each, built or refit by a SINGLE
//!      vkCmdBuildAccelerationStructuresKHR call carrying N build infos.
//!
//! Row layout inside one environment block is identical in both modes:
//!   row 0      ground quad (10 m x 10 m, identity rotation at the cell centre)
//!   rows 1..M  poseable rigid instances
//!   row M+1    marker slot: the isolation marker in env 0, a mask-0 dummy
//!              everywhere else (so instance counts stay uniform)

const std = @import("std");
const vk = @import("vk");
const gpu = @import("gpu.zig");
const meshes = @import("meshes.zig");

pub const Mode = enum { a, b };
pub const Update = enum { rebuild, refit };

pub const Config = struct {
    mode: Mode = .a,
    update: Update = .rebuild,
    n: u32 = 256,
    m: u32 = 8,
    res: u32 = 128,
    frames: u32 = 60,
    warmup: u32 = 10,
    spacing: f32 = 16.0,
    validation: bool = false,
};

/// Half extent of one environment's cell. The camera orbits at `orbit_radius` and
/// the ground quad is 10 m across, so the cell must contain both: 7 m leaves a
/// 1.5 m margin around the widest element and 2 m of empty space between cells at
/// the default 16 m spacing.
pub const cell_half = [3]f32{ 7.0, 7.0, 7.0 };
/// Camera orbit radius, always strictly inside `cell_half`.
pub const orbit_radius: f32 = 6.0;
/// Camera height above the ground plane.
pub const camera_height: f32 = 2.6;
/// Height of the camera's look-at target.
pub const target_height: f32 = 1.0;
/// Light direction, normalized, pointing from the surface toward the light.
pub const light_dir = [3]f32{ 0.42, 0.78, 0.46 };

/// Per-environment azimuth. A small spread keeps environments from being pixel
/// identical, so cross-environment bleed cannot hide behind symmetry.
pub fn cameraAzimuth(env: u32) f32 {
    return 0.35 + 0.013 * @as(f32, @floatFromInt(env % 17));
}

/// Camera eye position in environment-local space.
pub fn cameraEye(env: u32) [3]f32 {
    const az = cameraAzimuth(env);
    return .{ orbit_radius * @sin(az), camera_height, orbit_radius * @cos(az) };
}

pub const Status = enum {
    ok,
    skip_descriptor_limit,
    skip_instance_limit,
    skip_dispatch_limit,
    skip_output_range,
    skip_alloc,

    pub fn text(self: Status) []const u8 {
        return switch (self) {
            .ok => "ok",
            .skip_descriptor_limit => "skipped: acceleration-structure descriptor limit",
            .skip_instance_limit => "skipped: maxInstanceCount exceeded",
            .skip_dispatch_limit => "skipped: dispatch grid limit",
            .skip_output_range => "skipped: exceeds maxStorageBufferRange",
            .skip_alloc => "skipped: device allocation failed",
        };
    }
};

pub const Measurement = struct {
    as_ms_med: f64 = 0,
    as_ms_p90: f64 = 0,
    trace_ms_med: f64 = 0,
    trace_ms_p90: f64 = 0,
    wall_ms_med: f64 = 0,
    cpu_ms_med: f64 = 0,
    tlas_bytes: u64 = 0,
    blas_bytes: u64 = 0,
    instance_bytes: u64 = 0,
    fps: f64 = 0,
    status: Status = .ok,
    /// N * R * R packed pixels; only when the caller asks for them.
    images: ?[]u32 = null,
};

// ---------------------------------------------------------------------------
// Deterministic scene data
// ---------------------------------------------------------------------------

fn hash01(seed: u64) f32 {
    var x = seed *% 0x9E3779B97F4A7C15;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    x ^= x >> 31;
    // 24 bits is plenty for placement noise and is exact in f32.
    return @as(f32, @floatFromInt(x >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
}

/// `VkAccelerationStructureInstanceKHR` by hand: the C struct uses bitfields, so
/// translate-c emits it as an opaque type. Layout on a little-endian target is
/// 3x4 float transform, then {customIndex:24, mask:8}, then
/// {sbtRecordOffset:24, flags:8}, then the 8-byte BLAS reference: 64 bytes.
pub const Instance = extern struct {
    transform: [3][4]f32,
    custom_index_mask: u32,
    record_offset_flags: u32,
    acceleration_structure_reference: u64,

    comptime {
        std.debug.assert(@sizeOf(Instance) == 64);
        std.debug.assert(@offsetOf(Instance, "acceleration_structure_reference") == 56);
    }

    pub fn encode(self: Instance, custom_index: u32, mask: u8, flags: u32) Instance {
        var out = self;
        // Bitfield order on a little-endian target (LSB first, as the C bitfields
        // are declared): customIndex:24 then mask:8; sbtRecordOffset:24 then
        // flags:8.
        out.custom_index_mask = (custom_index & 0xFFFFFF) | (@as(u32, mask) << 24);
        out.record_offset_flags = (flags & 0xFF) << 24;
        return out;
    }
};

pub const instance_stride: usize = @sizeOf(Instance);

fn writeInstance(
    dst: [*]u8,
    index: usize,
    translation: [3]f32,
    scale: f32,
    yaw: f32,
    mask: u8,
    custom_index: u32,
    blas_address: vk.VkDeviceAddress,
) void {
    const row: *align(1) Instance = @ptrCast(dst + index * instance_stride);
    const c = @cos(yaw);
    const s = @sin(yaw);
    // Row-major 3x4: yaw about Y, then uniform scale.
    row.* = (Instance{
        .transform = .{
            .{ c * scale, 0, s * scale, translation[0] },
            .{ 0, scale, 0, translation[1] },
            .{ -s * scale, 0, c * scale, translation[2] },
        },
        .custom_index_mask = 0,
        .record_offset_flags = 0,
        .acceleration_structure_reference = blas_address,
    }).encode(custom_index, mask, vk.VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR);
}

/// Cell offset of environment `env` in a `side` x `side` centred grid.
pub fn cellOffset(env: u32, side: u32, spacing: f32) [3]f32 {
    const ix: f32 = @floatFromInt(env % side);
    const iz: f32 = @floatFromInt(env / side);
    const half = (@as(f32, @floatFromInt(side)) - 1.0) * 0.5;
    return .{ (ix - half) * spacing, 0, (iz - half) * spacing };
}

/// Smallest square grid that holds `n` environments.
pub fn gridSide(n: u32) u32 {
    var side: u32 = 1;
    while (side * side < n) side += 1;
    return side;
}

/// Environment-local transform of rigid instance `k` at frame `f`. A pure
/// function of (env, k, f), so two runs with the same frame count see exactly
/// the same geometry.
pub fn rigidTransform(env: u32, k: u32, frame: u32) struct { translation: [3]f32, scale: f32, yaw: f32, mesh: u32 } {
    const poseable: u32 = meshes.mesh_count - 4; // ids 0..11
    const seed = @as(u64, env) * 8191 + @as(u64, k) * 131 + @as(u64, frame) * 7919;
    const ang = 6.2831853 * hash01(seed);
    const rad = 2.2 + 1.6 * hash01(seed ^ 0x1234);
    const y = 0.55 + 1.5 * hash01(seed ^ 0x5678);
    const yaw = 6.2831853 * hash01(seed ^ 0x9ABC);
    const scale = 0.7 + 0.6 * hash01(seed ^ 0xDEF0);
    const mesh: u32 = @as(u32, @intFromFloat(hash01(seed ^ 0x2468) * @as(f32, @floatFromInt(poseable)))) % poseable;
    return .{ .translation = .{ rad * @cos(ang), y, rad * @sin(ang) }, .scale = scale, .yaw = yaw, .mesh = mesh };
}

/// Instances rows per environment: ground + M rigid + one marker slot.
pub fn rowsPerEnv(cfg: Config) u64 {
    return @as(u64, cfg.m) + 2;
}

fn fillInstances(dst: [*]u8, cfg: Config, side: u32, blas: *const meshes.BlasSet, frame: u32) void {
    const per_env = rowsPerEnv(cfg);
    const m: u64 = cfg.m;
    var env: u32 = 0;
    while (env < cfg.n) : (env += 1) {
        const off: [3]f32 = if (cfg.mode == .a) cellOffset(env, side, cfg.spacing) else .{ 0, 0, 0 };
        const base: usize = @intCast(@as(u64, env) * per_env);

        // Ground.
        writeInstance(dst, base, off, 1.0, 0.0, 0xFF, meshes.ground_mesh_id, blas.as[meshes.ground_mesh_id].address);

        var k: u32 = 0;
        while (k < m) : (k += 1) {
            const t = rigidTransform(env, k, frame);
            writeInstance(
                dst,
                base + 1 + k,
                .{ t.translation[0] + off[0], t.translation[1], t.translation[2] + off[2] },
                t.scale,
                t.yaw,
                0xFF,
                t.mesh,
                blas.as[t.mesh].address,
            );
        }

        // Marker slot. Env 0 gets an unlit sphere two metres in front of its own
        // camera along the view direction, so it is on screen and in front of the
        // rigid bodies. Every other env gets the same row with a zero ray mask,
        // which makes it untraversable, keeping the instance counts uniform.
        const is_marker = env == 0;
        const cam = cameraEye(env);
        const to_target = [3]f32{ -cam[0], target_height - cam[1], -cam[2] };
        const inv_len = 1.0 / @sqrt(to_target[0] * to_target[0] + to_target[1] * to_target[1] + to_target[2] * to_target[2]);
        const marker_pos = [3]f32{
            off[0] + cam[0] + to_target[0] * inv_len * 2.0,
            cam[1] + to_target[1] * inv_len * 2.0,
            off[2] + cam[2] + to_target[2] * inv_len * 2.0,
        };
        writeInstance(
            dst,
            base + @as(usize, @intCast(m + 1)),
            marker_pos,
            if (is_marker) 0.45 else 0.0,
            0.0,
            if (is_marker) 0xFF else 0x00,
            meshes.marker_mesh_id,
            blas.as[meshes.marker_mesh_id].address,
        );
    }
}

// ---------------------------------------------------------------------------
// Per-config run
// ---------------------------------------------------------------------------

pub fn run(
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    io: std.Io,
    cfg: Config,
    blas: *const meshes.BlasSet,
    vertex_buffer: gpu.Buffer,
    index_buffer: gpu.Buffer,
    mesh_buffer: gpu.Buffer,
    keep_images: bool,
) !Measurement {
    var result = Measurement{};

    const n: u64 = cfg.n;
    const res: u64 = cfg.res;
    const per_env = rowsPerEnv(cfg);
    const total_rows = n * per_env;
    const instance_bytes: vk.VkDeviceSize = @intCast(total_rows * instance_stride);
    const image_bytes: vk.VkDeviceSize = @intCast(n * res * res * 4);

    // (0) Feasibility: report a skip instead of failing.
    const tlas_count: u32 = if (cfg.mode == .a) 1 else cfg.n;
    const rows_per_tlas: u32 = if (cfg.mode == .a) @intCast(total_rows) else @intCast(per_env);
    if (tlas_count > ctx.as_properties.maxDescriptorSetAccelerationStructures) {
        result.status = .skip_descriptor_limit;
        return result;
    }
    if (rows_per_tlas > ctx.as_properties.maxInstanceCount) {
        result.status = .skip_instance_limit;
        return result;
    }
    if (image_bytes > ctx.properties.limits.maxStorageBufferRange or
        instance_bytes > ctx.properties.limits.maxStorageBufferRange)
    {
        result.status = .skip_output_range;
        return result;
    }
    const groups_xy: u32 = @intCast((res + 7) / 8);
    if (groups_xy > ctx.properties.limits.maxComputeWorkGroupCount[0] or
        cfg.n > ctx.properties.limits.maxComputeWorkGroupCount[2])
    {
        result.status = .skip_dispatch_limit;
        return result;
    }

    const side = gridSide(cfg.n);
    const host_visible = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const device_local = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // (1) Instance staging buffer: the CPU rewrites it every frame.
    const instance_buffer = ctx.createBuffer(
        instance_bytes,
        vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
            vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        host_visible,
        true,
    ) catch |e| return statusFromError(&result, e);
    defer ctx.destroyBuffer(instance_buffer);

    const image_buffer = ctx.createBuffer(
        image_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        device_local,
        false,
    ) catch |e| return statusFromError(&result, e);
    defer ctx.destroyBuffer(image_buffer);

    // (2) TLASes.
    const tlas = try gpa.alloc(gpu.AccelerationStructure, tlas_count);
    defer gpa.free(tlas);
    var allocated: u32 = 0;
    defer for (tlas[0..allocated]) |a| ctx.destroyAccelerationStructure(a);

    const geometries = try gpa.alloc(vk.VkAccelerationStructureGeometryKHR, tlas_count);
    defer gpa.free(geometries);
    const build_infos = try gpa.alloc(vk.VkAccelerationStructureBuildGeometryInfoKHR, tlas_count);
    defer gpa.free(build_infos);
    const range_infos = try gpa.alloc(vk.VkAccelerationStructureBuildRangeInfoKHR, tlas_count);
    defer gpa.free(range_infos);
    const sizes = try gpa.alloc(vk.VkAccelerationStructureBuildSizesInfoKHR, tlas_count);
    defer gpa.free(sizes);
    const counts = try gpa.alloc(u32, tlas_count);
    defer gpa.free(counts);

    // Rebuild: PREFER_FAST_BUILD every frame. Refit: built once with
    // PREFER_FAST_TRACE + ALLOW_UPDATE, then MODE_UPDATE every frame.
    const build_flags: vk.VkBuildAccelerationStructureFlagsKHR = if (cfg.update == .refit)
        vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
    else
        vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;

    for (0..tlas_count) |i| {
        counts[i] = if (cfg.mode == .a) @intCast(total_rows) else @intCast(per_env);

        geometries[i] = std.mem.zeroes(vk.VkAccelerationStructureGeometryKHR);
        geometries[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometries[i].geometryType = vk.VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometries[i].flags = vk.VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometries[i].geometry.instances.sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        geometries[i].geometry.instances.arrayOfPointers = vk.VK_FALSE;
        geometries[i].geometry.instances.data = .{
            .deviceAddress = if (cfg.mode == .a)
                instance_buffer.address
            else
                instance_buffer.address + @as(vk.VkDeviceAddress, @intCast(i)) * per_env * instance_stride,
        };

        build_infos[i] = std.mem.zeroes(vk.VkAccelerationStructureBuildGeometryInfoKHR);
        build_infos[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_infos[i].type = vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        build_infos[i].flags = build_flags;
        build_infos[i].mode = vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_infos[i].geometryCount = 1;
        build_infos[i].pGeometries = &geometries[i];

        sizes[i] = std.mem.zeroes(vk.VkAccelerationStructureBuildSizesInfoKHR);
        sizes[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        ctx.vkGetAccelerationStructureBuildSizesKHR(
            ctx.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_infos[i],
            &counts[i],
            &sizes[i],
        );
        tlas[i] = ctx.createAccelerationStructure(vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes[i].accelerationStructureSize) catch |e| return statusFromError(&result, e);
        allocated += 1;
        result.tlas_bytes += sizes[i].accelerationStructureSize;
        build_infos[i].dstAccelerationStructure = tlas[i].handle;
    }

    // One scratch buffer, per-TLAS offsets aligned to the required alignment.
    const align_to: vk.VkDeviceSize = @max(ctx.as_properties.minAccelerationStructureScratchOffsetAlignment, 1);
    const scratch_offsets = try gpa.alloc(vk.VkDeviceSize, tlas_count);
    defer gpa.free(scratch_offsets);
    var scratch_bytes: vk.VkDeviceSize = 0;
    for (0..tlas_count) |i| {
        scratch_offsets[i] = std.mem.alignForward(vk.VkDeviceSize, scratch_bytes, align_to);
        scratch_bytes = scratch_offsets[i] + @max(sizes[i].buildScratchSize, sizes[i].updateScratchSize);
    }
    const scratch = ctx.createBuffer(
        scratch_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        device_local,
        false,
    ) catch |e| return statusFromError(&result, e);
    defer ctx.destroyBuffer(scratch);
    for (0..tlas_count) |i| {
        build_infos[i].scratchData = .{ .deviceAddress = scratch.address + scratch_offsets[i] };
        range_infos[i] = .{ .primitiveCount = counts[i], .primitiveOffset = 0, .firstVertex = 0, .transformOffset = 0 };
    }

    // (3) Descriptor set: AS array (variable count) + four storage buffers.
    var bindings: [5]vk.VkDescriptorSetLayoutBinding = undefined;
    bindings[0] = .{
        .binding = 0,
        .descriptorType = vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .descriptorCount = tlas_count,
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = null,
    };
    for (1..5) |b| {
        bindings[b] = .{
            .binding = @intCast(b),
            .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = null,
        };
    }

    var binding_flags: [5]vk.VkDescriptorBindingFlags = .{
        vk.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | vk.VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
        0,
        0,
        0,
        0,
    };
    var flags_info = std.mem.zeroes(vk.VkDescriptorSetLayoutBindingFlagsCreateInfo);
    flags_info.sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_info.bindingCount = bindings.len;
    flags_info.pBindingFlags = &binding_flags;

    var dsl_ci = std.mem.zeroes(vk.VkDescriptorSetLayoutCreateInfo);
    dsl_ci.sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_ci.pNext = &flags_info;
    dsl_ci.bindingCount = bindings.len;
    dsl_ci.pBindings = &bindings;
    var set_layout: vk.VkDescriptorSetLayout = null;
    try gpu.check(vk.vkCreateDescriptorSetLayout(ctx.device, &dsl_ci, null, &set_layout), "vkCreateDescriptorSetLayout");
    defer vk.vkDestroyDescriptorSetLayout(ctx.device, set_layout, null);

    const pool_sizes = [_]vk.VkDescriptorPoolSize{
        .{ .type = vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = tlas_count },
        .{ .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 4 },
    };
    var pool_ci = std.mem.zeroes(vk.VkDescriptorPoolCreateInfo);
    pool_ci.sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_ci.maxSets = 1;
    pool_ci.poolSizeCount = pool_sizes.len;
    pool_ci.pPoolSizes = &pool_sizes;
    var pool: vk.VkDescriptorPool = null;
    try gpu.check(vk.vkCreateDescriptorPool(ctx.device, &pool_ci, null, &pool), "vkCreateDescriptorPool");
    defer vk.vkDestroyDescriptorPool(ctx.device, pool, null);

    const var_counts = [_]u32{tlas_count};
    var variable_count = std.mem.zeroes(vk.VkDescriptorSetVariableDescriptorCountAllocateInfo);
    variable_count.sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    variable_count.descriptorSetCount = 1;
    variable_count.pDescriptorCounts = &var_counts;

    var alloc_ci = std.mem.zeroes(vk.VkDescriptorSetAllocateInfo);
    alloc_ci.sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_ci.pNext = &variable_count;
    alloc_ci.descriptorPool = pool;
    alloc_ci.descriptorSetCount = 1;
    alloc_ci.pSetLayouts = &set_layout;
    var desc_set: vk.VkDescriptorSet = null;
    try gpu.check(vk.vkAllocateDescriptorSets(ctx.device, &alloc_ci, &desc_set), "vkAllocateDescriptorSets");

    const as_handles = try gpa.alloc(vk.VkAccelerationStructureKHR, tlas_count);
    defer gpa.free(as_handles);
    for (0..tlas_count) |i| as_handles[i] = tlas[i].handle;

    var write_as = std.mem.zeroes(vk.VkWriteDescriptorSetAccelerationStructureKHR);
    write_as.sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    write_as.accelerationStructureCount = tlas_count;
    write_as.pAccelerationStructures = as_handles.ptr;

    var buffer_infos = [_]vk.VkDescriptorBufferInfo{
        .{ .buffer = image_buffer.buffer, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = vertex_buffer.buffer, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = index_buffer.buffer, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = mesh_buffer.buffer, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    var writes: [5]vk.VkWriteDescriptorSet = undefined;
    writes[0] = .{
        .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &write_as,
        .dstSet = desc_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = tlas_count,
        .descriptorType = vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        .pImageInfo = null,
        .pBufferInfo = null,
        .pTexelBufferView = null,
    };
    for (1..5) |b| {
        writes[b] = .{
            .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = desc_set,
            .dstBinding = @intCast(b),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = null,
            .pBufferInfo = &buffer_infos[b - 1],
            .pTexelBufferView = null,
        };
    }
    vk.vkUpdateDescriptorSets(ctx.device, writes.len, &writes, 0, null);

    // (4) Pipeline.
    const spv = @embedFile("trace.spv");
    var shader_ci = std.mem.zeroes(vk.VkShaderModuleCreateInfo);
    shader_ci.sType = vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_ci.codeSize = spv.len;
    shader_ci.pCode = @ptrCast(@alignCast(spv.ptr));
    var shader: vk.VkShaderModule = null;
    try gpu.check(vk.vkCreateShaderModule(ctx.device, &shader_ci, null, &shader), "vkCreateShaderModule");
    defer vk.vkDestroyShaderModule(ctx.device, shader, null);

    var push_range = std.mem.zeroes(vk.VkPushConstantRange);
    push_range.stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = 4 * 16;

    var layout_ci = std.mem.zeroes(vk.VkPipelineLayoutCreateInfo);
    layout_ci.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_ci.setLayoutCount = 1;
    layout_ci.pSetLayouts = &set_layout;
    layout_ci.pushConstantRangeCount = 1;
    layout_ci.pPushConstantRanges = &push_range;
    var pipeline_layout: vk.VkPipelineLayout = null;
    try gpu.check(vk.vkCreatePipelineLayout(ctx.device, &layout_ci, null, &pipeline_layout), "vkCreatePipelineLayout");
    defer vk.vkDestroyPipelineLayout(ctx.device, pipeline_layout, null);

    var stage = std.mem.zeroes(vk.VkPipelineShaderStageCreateInfo);
    stage.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = vk.VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = "main";

    var pipeline_ci = std.mem.zeroes(vk.VkComputePipelineCreateInfo);
    pipeline_ci.sType = vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_ci.stage = stage;
    pipeline_ci.layout = pipeline_layout;
    var pipeline: vk.VkPipeline = null;
    try gpu.check(vk.vkCreateComputePipelines(ctx.device, null, 1, &pipeline_ci, null, &pipeline), "vkCreateComputePipelines");
    defer vk.vkDestroyPipeline(ctx.device, pipeline, null);

    var query_ci = std.mem.zeroes(vk.VkQueryPoolCreateInfo);
    query_ci.sType = vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_ci.queryType = vk.VK_QUERY_TYPE_TIMESTAMP;
    query_ci.queryCount = 4;
    var query_pool: vk.VkQueryPool = null;
    try gpu.check(vk.vkCreateQueryPool(ctx.device, &query_ci, null, &query_pool), "vkCreateQueryPool");
    defer vk.vkDestroyQueryPool(ctx.device, query_pool, null);

    // (5) Frame loop.
    const as_times = try gpa.alloc(f64, cfg.frames);
    defer gpa.free(as_times);
    const trace_times = try gpa.alloc(f64, cfg.frames);
    defer gpa.free(trace_times);
    const wall_times = try gpa.alloc(f64, cfg.frames);
    defer gpa.free(wall_times);
    const cpu_times = try gpa.alloc(f64, cfg.frames);
    defer gpa.free(cpu_times);

    const range_ptrs = try gpa.alloc([*c]const vk.VkAccelerationStructureBuildRangeInfoKHR, tlas_count);
    defer gpa.free(range_ptrs);
    for (0..tlas_count) |i| range_ptrs[i] = &range_infos[i];

    var pc: [4][4]f32 = .{
        .{ @floatFromInt(side), @floatFromInt(side), @floatFromInt(res), if (cfg.mode == .a) 1 else 0 },
        .{ cell_half[0], cell_half[1], cell_half[2], 0 },
        .{ cfg.spacing, orbit_radius, camera_height, target_height },
        .{ light_dir[0], light_dir[1], light_dir[2], 0 },
    };

    const timestamp_period: f64 = ctx.properties.limits.timestampPeriod;
    const frames_total = cfg.warmup + cfg.frames;
    var frame: u32 = 0;
    while (frame < frames_total) : (frame += 1) {
        // (1) CPU jitter. Excluded from GPU timings but measured on its own.
        const cpu_t0 = std.Io.Clock.awake.now(io);
        fillInstances(instance_buffer.mapped.?, cfg, side, blas, frame);
        const cpu_t1 = std.Io.Clock.awake.now(io);

        // (2)+(3) AS update and one dispatch over all N cameras.
        try ctx.beginCommands();
        vk.vkCmdResetQueryPool(ctx.cmd, query_pool, 0, 4);
        vk.vkCmdWriteTimestamp(ctx.cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 0);

        const build_mode: vk.VkBuildAccelerationStructureModeKHR = if (cfg.update == .refit and frame != 0)
            vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
        else
            vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        for (0..tlas_count) |i| {
            build_infos[i].mode = build_mode;
            build_infos[i].srcAccelerationStructure = if (build_mode == vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) tlas[i].handle else null;
        }
        ctx.vkCmdBuildAccelerationStructuresKHR(ctx.cmd, tlas_count, build_infos.ptr, range_ptrs.ptr);

        var barrier = std.mem.zeroes(vk.VkMemoryBarrier);
        barrier.sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vk.vkCmdPipelineBarrier(
            ctx.cmd,
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
        vk.vkCmdWriteTimestamp(ctx.cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 1);

        vk.vkCmdBindPipeline(ctx.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vk.vkCmdBindDescriptorSets(ctx.cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, null);
        vk.vkCmdPushConstants(ctx.cmd, pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * 16, &pc);
        vk.vkCmdDispatch(ctx.cmd, groups_xy, groups_xy, cfg.n);
        vk.vkCmdWriteTimestamp(ctx.cmd, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, query_pool, 2);
        try ctx.endCommands();

        const wall_t0 = std.Io.Clock.awake.now(io);
        try ctx.submitAndWait();
        const wall_t1 = std.Io.Clock.awake.now(io);

        if (frame < cfg.warmup) continue;

        const slot = frame - cfg.warmup;
        var stamps = [3]u64{ 0, 0, 0 };
        try gpu.check(
            vk.vkGetQueryPoolResults(ctx.device, query_pool, 0, 3, @sizeOf([3]u64), &stamps, @sizeOf(u64), vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT),
            "vkGetQueryPoolResults",
        );
        as_times[slot] = @as(f64, @floatFromInt(stamps[1] -% stamps[0])) * timestamp_period;
        trace_times[slot] = @as(f64, @floatFromInt(stamps[2] -% stamps[1])) * timestamp_period;
        wall_times[slot] = @as(f64, @floatFromInt(wall_t0.durationTo(wall_t1).nanoseconds));
        cpu_times[slot] = @as(f64, @floatFromInt(cpu_t0.durationTo(cpu_t1).nanoseconds));
    }

    // (6) Aggregate: median and p90 over the measured frames.
    std.mem.sortUnstable(f64, as_times, {}, comptime std.sort.asc(f64));
    std.mem.sortUnstable(f64, trace_times, {}, comptime std.sort.asc(f64));
    std.mem.sortUnstable(f64, wall_times, {}, comptime std.sort.asc(f64));
    std.mem.sortUnstable(f64, cpu_times, {}, comptime std.sort.asc(f64));

    // All four series are nanoseconds; report milliseconds.
    result.as_ms_med = percentile(as_times, 0.5) / 1.0e6;
    result.as_ms_p90 = percentile(as_times, 0.9) / 1.0e6;
    result.trace_ms_med = percentile(trace_times, 0.5) / 1.0e6;
    result.trace_ms_p90 = percentile(trace_times, 0.9) / 1.0e6;
    result.wall_ms_med = percentile(wall_times, 0.5) / 1.0e6;
    result.cpu_ms_med = percentile(cpu_times, 0.5) / 1.0e6;
    result.blas_bytes = blas.bytes;
    result.instance_bytes = instance_bytes;

    const e2e_ms = result.as_ms_med + result.trace_ms_med;
    result.fps = if (e2e_ms > 0) @as(f64, @floatFromInt(cfg.n)) / (e2e_ms / 1000.0) else 0;

    if (keep_images) {
        const images = try gpa.alloc(u32, @intCast(n * res * res));
        errdefer gpa.free(images);
        const staging = ctx.createBuffer(image_bytes, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host_visible, true) catch |e| {
            gpa.free(images);
            return e;
        };
        defer ctx.destroyBuffer(staging);
        try ctx.beginCommands();
        var copy = std.mem.zeroes(vk.VkBufferCopy);
        copy.size = image_bytes;
        vk.vkCmdCopyBuffer(ctx.cmd, image_buffer.buffer, staging.buffer, 1, &copy);
        try ctx.endCommands();
        try ctx.submitAndWait();
        @memcpy(std.mem.sliceAsBytes(images), staging.mapped.?[0..@intCast(image_bytes)]);
        result.images = images;
    }
    return result;
}

fn statusFromError(result: *Measurement, e: anyerror) anyerror!Measurement {
    result.status = .skip_alloc;
    if (e == error.OutOfDeviceMemory or e == error.NoSuitableMemoryType) return result.*;
    return e;
}

fn percentile(sorted: []const f64, p: f64) f64 {
    if (sorted.len == 0) return 0;
    if (sorted.len == 1) return sorted[0];
    const pos = p * @as(f64, @floatFromInt(sorted.len - 1));
    const lo: usize = @intFromFloat(@floor(pos));
    const hi = @min(lo + 1, sorted.len - 1);
    const frac = pos - @as(f64, @floatFromInt(lo));
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}
