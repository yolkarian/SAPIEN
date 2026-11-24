//! Procedural triangle meshes and the shared BLAS set.
//!
//! Every environment reuses the same K BLASes: 12 subdivided rigid-body meshes,
//! one large ground quad, and 3 extra meshes (one reserved as the isolation
//! marker). Mesh id equals BLAS index.

const std = @import("std");
const vk = @import("vk");
const gpu = @import("gpu.zig");

pub const mesh_count: u32 = 16;
pub const ground_mesh_id: u32 = 12;
pub const marker_mesh_id: u32 = 15;

/// Half extent of the ground quad, metres.
pub const ground_half: f32 = 5.0;

pub const MeshRange = struct {
    first_index: u32,
    index_count: u32,
    first_vertex: u32,
    vertex_count: u32,
};

pub const MeshSet = struct {
    /// 4 floats per vertex (xyz + 1.0).
    vertices: std.ArrayList(f32) = .empty,
    indices: std.ArrayList(u32) = .empty,
    ranges: [mesh_count]MeshRange = undefined,
    /// Mesh currently being appended, so indices can be stored mesh-local.
    current_mesh: u32 = 0,

    pub fn deinit(self: *MeshSet, gpa: std.mem.Allocator) void {
        self.vertices.deinit(gpa);
        self.indices.deinit(gpa);
    }

    fn pushVertex(self: *MeshSet, gpa: std.mem.Allocator, x: f32, y: f32, z: f32) !u32 {
        const idx: u32 = @intCast(self.vertices.items.len / 4);
        try self.vertices.appendSlice(gpa, &.{ x, y, z, 1.0 });
        return idx;
    }

    /// `a`, `b`, `c` are absolute vertex indices; they are stored relative to
    /// the current mesh's first vertex, which is what both the BLAS build and the
    /// shader expect (indices are always mesh-local).
    fn pushTri(self: *MeshSet, gpa: std.mem.Allocator, a: u32, b: u32, c: u32) !void {
        const base = self.ranges[self.current_mesh].first_vertex;
        try self.indices.appendSlice(gpa, &.{ a - base, b - base, c - base });
    }

    fn begin(self: *MeshSet, mesh: u32) void {
        self.current_mesh = mesh;
        self.ranges[mesh] = .{
            .first_index = @intCast(self.indices.items.len),
            .index_count = 0,
            .first_vertex = @intCast(self.vertices.items.len / 4),
            .vertex_count = 0,
        };
    }

    fn end(self: *MeshSet, mesh: u32) void {
        const r = &self.ranges[mesh];
        r.index_count = @intCast(self.indices.items.len - r.first_index);
        r.vertex_count = @intCast(self.vertices.items.len / 4 - r.first_vertex);
    }

    /// UV sphere; `stacks * slices * 2` triangles minus the two pole fans'
    /// degenerate rows.
    pub fn addSphere(self: *MeshSet, gpa: std.mem.Allocator, mesh: u32, radius: f32, stacks: u32, slices: u32) !void {
        self.begin(mesh);
        const base: u32 = @intCast(self.vertices.items.len / 4);
        const north = try self.pushVertex(gpa, 0, radius, 0);
        // Interior latitude rings, 1..stacks-1.
        for (1..stacks) |i| {
            const theta = std.math.pi * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(stacks));
            const st = @sin(theta);
            const ct = @cos(theta);
            for (0..slices) |j| {
                const phi = 2.0 * std.math.pi * @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(slices));
                _ = try self.pushVertex(gpa, radius * st * @cos(phi), radius * ct, radius * st * @sin(phi));
            }
        }
        const south = try self.pushVertex(gpa, 0, -radius, 0);

        const ring = struct {
            fn at(b: u32, n_slices: u32, i: usize, j: usize) u32 {
                return b + 1 + @as(u32, @intCast((i - 1) * n_slices + (j % n_slices)));
            }
        }.at;

        for (0..slices) |j| {
            try self.pushTri(gpa, north, ring(base, slices, 1, j), ring(base, slices, 1, j + 1));
        }
        for (1..stacks - 1) |i| {
            for (0..slices) |j| {
                const a = ring(base, slices, i, j);
                const b = ring(base, slices, i, j + 1);
                const c = ring(base, slices, i + 1, j);
                const d = ring(base, slices, i + 1, j + 1);
                try self.pushTri(gpa, a, c, b);
                try self.pushTri(gpa, b, c, d);
            }
        }
        for (0..slices) |j| {
            try self.pushTri(gpa, south, ring(base, slices, stacks - 1, j + 1), ring(base, slices, stacks - 1, j));
        }
        self.end(mesh);
    }

    /// Axis-aligned box, `n` segments per edge, six independent faces.
    pub fn addBox(self: *MeshSet, gpa: std.mem.Allocator, mesh: u32, extent: [3]f32, n: u32) !void {
        self.begin(mesh);
        const half = [3]f32{ extent[0] * 0.5, extent[1] * 0.5, extent[2] * 0.5 };
        const axes = [6]struct { axis: u3, sign: f32, u: u3, v: u3 }{
            .{ .axis = 0, .sign = 1, .u = 1, .v = 2 },
            .{ .axis = 0, .sign = -1, .u = 2, .v = 1 },
            .{ .axis = 1, .sign = 1, .u = 2, .v = 0 },
            .{ .axis = 1, .sign = -1, .u = 0, .v = 2 },
            .{ .axis = 2, .sign = 1, .u = 0, .v = 1 },
            .{ .axis = 2, .sign = -1, .u = 1, .v = 0 },
        };
        const segs = n + 1;
        for (axes) |face| {
            const base: u32 = @intCast(self.vertices.items.len / 4);
            for (0..segs) |iv| {
                for (0..segs) |iu| {
                    const fu = @as(f32, @floatFromInt(iu)) / @as(f32, @floatFromInt(n)) * 2.0 - 1.0;
                    const fv = @as(f32, @floatFromInt(iv)) / @as(f32, @floatFromInt(n)) * 2.0 - 1.0;
                    var p = [3]f32{ 0, 0, 0 };
                    p[face.axis] = face.sign * half[face.axis];
                    p[face.u] = fu * half[face.u];
                    p[face.v] = fv * half[face.v];
                    _ = try self.pushVertex(gpa, p[0], p[1], p[2]);
                }
            }
            for (0..n) |iv| {
                for (0..n) |iu| {
                    const a = base + @as(u32, @intCast(iv * segs + iu));
                    const b = a + 1;
                    const c = a + @as(u32, @intCast(segs));
                    const d = c + 1;
                    try self.pushTri(gpa, a, b, c);
                    try self.pushTri(gpa, b, d, c);
                }
            }
        }
        self.end(mesh);
    }

    /// Flat quad in the xz plane at y = 0.
    pub fn addGround(self: *MeshSet, gpa: std.mem.Allocator, mesh: u32, half: f32) !void {
        self.begin(mesh);
        const a = try self.pushVertex(gpa, -half, 0, -half);
        const b = try self.pushVertex(gpa, half, 0, -half);
        const c = try self.pushVertex(gpa, half, 0, half);
        const d = try self.pushVertex(gpa, -half, 0, half);
        try self.pushTri(gpa, a, b, c);
        try self.pushTri(gpa, a, c, d);
        self.end(mesh);
    }

    pub fn buildAll(self: *MeshSet, gpa: std.mem.Allocator) !void {
        // Eight subdivided spheres, ~1k triangles each.
        try self.addSphere(gpa, 0, 0.55, 24, 24);
        try self.addSphere(gpa, 1, 0.40, 18, 28);
        try self.addSphere(gpa, 2, 0.70, 26, 22);
        try self.addSphere(gpa, 3, 0.32, 16, 24);
        try self.addSphere(gpa, 4, 0.62, 22, 26);
        try self.addSphere(gpa, 5, 0.48, 20, 24);
        try self.addSphere(gpa, 6, 0.58, 24, 20);
        try self.addSphere(gpa, 7, 0.44, 18, 26);
        // Four subdivided boxes, ~1k triangles each.
        try self.addBox(gpa, 8, .{ 1.0, 0.7, 0.5 }, 12);
        try self.addBox(gpa, 9, .{ 0.6, 1.1, 0.6 }, 12);
        try self.addBox(gpa, 10, .{ 1.3, 0.4, 0.8 }, 12);
        try self.addBox(gpa, 11, .{ 0.5, 0.5, 1.2 }, 12);
        try self.addGround(gpa, ground_mesh_id, ground_half);
        try self.addSphere(gpa, 13, 0.90, 20, 20);
        try self.addSphere(gpa, 14, 0.25, 12, 16);
        try self.addSphere(gpa, marker_mesh_id, 1.0, 16, 16);
    }

    pub fn triangleCount(self: *const MeshSet) u32 {
        return @intCast(self.indices.items.len / 3);
    }

    pub fn vertexCount(self: *const MeshSet) u32 {
        return @intCast(self.vertices.items.len / 4);
    }
};

/// The BLAS set shared by both modes and by every environment.
pub const BlasSet = struct {
    as: [mesh_count]gpu.AccelerationStructure = undefined,
    bytes: u64 = 0,

    pub fn destroy(self: *BlasSet, ctx: *gpu.Context) void {
        for (&self.as) |*a| ctx.destroyAccelerationStructure(a.*);
    }
};

/// Builds all K BLASes in one recorded build batch.
pub fn buildBlases(
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    meshes: *const MeshSet,
    vertex_buffer: gpu.Buffer,
    index_buffer: gpu.Buffer,
) !BlasSet {
    var out = BlasSet{};

    const geometry = try gpa.alloc(vk.VkAccelerationStructureGeometryKHR, mesh_count);
    defer gpa.free(geometry);
    const build_info = try gpa.alloc(vk.VkAccelerationStructureBuildGeometryInfoKHR, mesh_count);
    defer gpa.free(build_info);
    const range_info = try gpa.alloc(vk.VkAccelerationStructureBuildRangeInfoKHR, mesh_count);
    defer gpa.free(range_info);
    const sizes = try gpa.alloc(vk.VkAccelerationStructureBuildSizesInfoKHR, mesh_count);
    defer gpa.free(sizes);
    const prim_counts = try gpa.alloc(u32, mesh_count);
    defer gpa.free(prim_counts);

    for (0..mesh_count) |i| {
        const r = meshes.ranges[i];
        prim_counts[i] = r.index_count / 3;

        geometry[i] = std.mem.zeroes(vk.VkAccelerationStructureGeometryKHR);
        geometry[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry[i].geometryType = vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry[i].flags = vk.VK_GEOMETRY_OPAQUE_BIT_KHR;
        const tri = &geometry[i].geometry.triangles;
        tri.sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = vk.VK_FORMAT_R32G32B32A32_SFLOAT;
        tri.vertexData = .{ .deviceAddress = vertex_buffer.address + @as(vk.VkDeviceAddress, r.first_vertex) * 16 };
        tri.vertexStride = 16;
        tri.maxVertex = r.vertex_count - 1;
        tri.indexType = vk.VK_INDEX_TYPE_UINT32;
        tri.indexData = .{ .deviceAddress = index_buffer.address + @as(vk.VkDeviceAddress, r.first_index) * 4 };

        build_info[i] = std.mem.zeroes(vk.VkAccelerationStructureBuildGeometryInfoKHR);
        build_info[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        build_info[i].type = vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        build_info[i].flags = vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build_info[i].mode = vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build_info[i].geometryCount = 1;
        build_info[i].pGeometries = &geometry[i];

        sizes[i] = std.mem.zeroes(vk.VkAccelerationStructureBuildSizesInfoKHR);
        sizes[i].sType = vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        ctx.vkGetAccelerationStructureBuildSizesKHR(
            ctx.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &build_info[i],
            &prim_counts[i],
            &sizes[i],
        );
        out.as[i] = try ctx.createAccelerationStructure(vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes[i].accelerationStructureSize);
        out.bytes += sizes[i].accelerationStructureSize;
        build_info[i].dstAccelerationStructure = out.as[i].handle;
    }

    // One scratch allocation with per-BLAS aligned offsets.
    const align_to: vk.VkDeviceSize = @max(ctx.as_properties.minAccelerationStructureScratchOffsetAlignment, 1);
    const scratch_offsets = try gpa.alloc(vk.VkDeviceSize, mesh_count);
    defer gpa.free(scratch_offsets);
    var scratch_bytes: vk.VkDeviceSize = 0;
    for (0..mesh_count) |i| {
        scratch_offsets[i] = std.mem.alignForward(vk.VkDeviceSize, scratch_bytes, align_to);
        scratch_bytes = scratch_offsets[i] + sizes[i].buildScratchSize;
    }
    const scratch = try ctx.createBuffer(
        scratch_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        false,
    );
    defer ctx.destroyBuffer(scratch);

    for (0..mesh_count) |i| {
        build_info[i].scratchData = .{ .deviceAddress = scratch.address + scratch_offsets[i] };
        range_info[i] = .{ .primitiveCount = prim_counts[i], .primitiveOffset = 0, .firstVertex = 0, .transformOffset = 0 };
    }

    const pointers = try gpa.alloc([*c]const vk.VkAccelerationStructureBuildRangeInfoKHR, mesh_count);
    defer gpa.free(pointers);
    for (0..mesh_count) |i| pointers[i] = &range_info[i];

    try ctx.beginCommands();
    ctx.vkCmdBuildAccelerationStructuresKHR(ctx.cmd, mesh_count, build_info.ptr, pointers.ptr);
    try ctx.endCommands();
    try ctx.submitAndWait();

    // Make the built BLASes visible to later builds.
    try ctx.beginCommands();
    var barrier = std.mem.zeroes(vk.VkMemoryBarrier);
    barrier.sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vk.vkCmdPipelineBarrier(
        ctx.cmd,
        vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0,
        1,
        &barrier,
        0,
        null,
        0,
        null,
    );
    try ctx.endCommands();
    try ctx.submitAndWait();
    return out;
}
