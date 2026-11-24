//! TLAS isolation benchmark: argument parsing, matrix sweep, correctness guard.
//!
//! Two schemes are compared for isolating N environments sharing K BLASes:
//!   A  one TLAS + per-environment spatial offset, rays clipped to the cell
//!   B  one TLAS per environment in a descriptor array, indexed per env
//!
//! The measured quantity is end-to-end GPU time per frame: AS update (rebuild or
//! refit) plus one compute dispatch that traces N cameras.

const std = @import("std");
const vk = @import("vk");
const gpu = @import("gpu.zig");
const meshes = @import("meshes.zig");
const scene = @import("scene.zig");

const csv_header = "mode,update,N,M,R,as_ms_med,as_ms_p90,trace_ms_med,trace_ms_p90,wall_ms_med,cpu_ms_med,tlas_bytes,blas_bytes,instance_bytes,fps_images_per_s,status\n";

pub const Options = struct {
    cfg: scene.Config = .{},
    matrix: bool = false,
    verify: bool = false,
    single: bool = false,
    validation: bool = false,
    csv_path: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;

    var args = std.ArrayList([]const u8).empty;
    defer args.deinit(gpa);
    var arg_it = init.minimal.args.iterate();
    while (arg_it.next()) |a| try args.append(gpa, a);

    const opts = try parseArgs(args.items);
    if (opts == null) return;

    var ctx = try gpu.Context.init(opts.?.validation);
    defer ctx.deinit();

    std.debug.print("heap(device-local)={d:.2} GiB  maxInstanceCount={d}  maxDescriptorSetAS={d}\n", .{
        @as(f64, @floatFromInt(ctx.deviceLocalHeapBytes())) / (1024.0 * 1024.0 * 1024.0),
        ctx.as_properties.maxInstanceCount,
        ctx.as_properties.maxDescriptorSetAccelerationStructures,
    });

    // Shared geometry: procedural meshes and their BLASes.
    var mset = meshes.MeshSet{};
    defer mset.deinit(gpa);
    try mset.buildAll(gpa);
    std.debug.print("meshes: {d} BLASes, {d} vertices, {d} triangles\n", .{
        meshes.mesh_count, mset.vertexCount(), mset.triangleCount(),
    });

    const host_visible = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const device_local = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const shared_usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    const vb = try ctx.createBuffer(@intCast(mset.vertices.items.len * @sizeOf(f32)), shared_usage, device_local, false);
    defer ctx.destroyBuffer(vb);
    const ib = try ctx.createBuffer(@intCast(mset.indices.items.len * @sizeOf(u32)), shared_usage, device_local, false);
    defer ctx.destroyBuffer(ib);

    var ranges: [meshes.mesh_count][2]u32 = undefined;
    for (0..meshes.mesh_count) |k| ranges[k] = .{ mset.ranges[k].first_index, mset.ranges[k].first_vertex };
    const mb = try ctx.createBuffer(@intCast(@sizeOf(@TypeOf(ranges))), shared_usage, device_local, false);
    defer ctx.destroyBuffer(mb);

    // One-shot upload of the three shared buffers.
    {
        const v_bytes: usize = mset.vertices.items.len * @sizeOf(f32);
        const i_bytes: usize = mset.indices.items.len * @sizeOf(u32);
        const m_bytes: usize = @sizeOf(@TypeOf(ranges));
        const vs = try ctx.createBuffer(@intCast(v_bytes), vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_visible, true);
        defer ctx.destroyBuffer(vs);
        const is = try ctx.createBuffer(@intCast(i_bytes), vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_visible, true);
        defer ctx.destroyBuffer(is);
        const ms = try ctx.createBuffer(@intCast(m_bytes), vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, host_visible, true);
        defer ctx.destroyBuffer(ms);

        @memcpy(vs.mapped.?[0..v_bytes], std.mem.sliceAsBytes(mset.vertices.items));
        @memcpy(is.mapped.?[0..i_bytes], std.mem.sliceAsBytes(mset.indices.items));
        @memcpy(ms.mapped.?[0..m_bytes], std.mem.sliceAsBytes(&ranges));

        try ctx.beginCommands();
        var r0 = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @intCast(v_bytes) };
        var r1 = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @intCast(i_bytes) };
        var r2 = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @intCast(m_bytes) };
        vk.vkCmdCopyBuffer(ctx.cmd, vs.buffer, vb.buffer, 1, &r0);
        vk.vkCmdCopyBuffer(ctx.cmd, is.buffer, ib.buffer, 1, &r1);
        vk.vkCmdCopyBuffer(ctx.cmd, ms.buffer, mb.buffer, 1, &r2);
        var barrier = std.mem.zeroes(vk.VkMemoryBarrier);
        barrier.sType = vk.VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vk.vkCmdPipelineBarrier(
            ctx.cmd,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | vk.VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
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
    }

    var blas_set = try meshes.buildBlases(&ctx, gpa, &mset, vb, ib);
    defer blas_set.destroy(&ctx);
    std.debug.print("BLAS: {d} bytes over {d} structures\n", .{ blas_set.bytes, meshes.mesh_count });

    if (opts.?.verify) {
        const pass = runCorrectnessGuard(&ctx, gpa, io, opts.?.cfg, &blas_set, vb, ib, mb);
        if (!(try pass)) return error.CorrectnessGuardFailed;
    }

    if (opts.?.matrix) {
        try runMatrix(&ctx, gpa, io, opts.?.csv_path, &blas_set, vb, ib, mb);
    } else if (!opts.?.verify) {
        var r = try scene.run(&ctx, gpa, io, opts.?.cfg, &blas_set, vb, ib, mb, false);
        try printSummary(opts.?.cfg, &r);
        try emitCsvRow(io, opts.?.csv_path, opts.?.cfg, &r, true);
    }
}

const usage =
    \\usage: tlas_isolation [options]
    \\
    \\  --matrix                  run the full N x M x R x mode x update sweep
    \\  --verify                  run the correctness guard (N=16, M=8, R=64)
    \\  --mode A|B                isolation scheme (default A)
    \\  --update rebuild|refit    TLAS update strategy (default rebuild)
    \\  --n N                     environments (default 256)
    \\  --m M                     rigid instances per environment (default 8)
    \\  --res R                   square camera resolution (default 128)
    \\  --frames F                measured frames after warmup (default 60)
    \\  --warmup F                warmup frames (default 10)
    \\  --spacing S               mode A grid spacing in metres (default 16)
    \\  --csv PATH                append CSV rows here ('-' = stdout)
    \\  --validation              enable validation layers when installed
    \\  -h, --help                show this message
    \\
;

fn parseArgs(args: []const []const u8) !?Options {
    var opts = Options{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--matrix")) {
            opts.matrix = true;
            opts.single = true;
        } else if (std.mem.eql(u8, a, "--verify")) {
            opts.verify = true;
        } else if (std.mem.eql(u8, a, "--validation")) {
            opts.validation = true;
        } else if (std.mem.eql(u8, a, "--mode")) {
            i += 1;
            if (i >= args.len) return badArg("--mode needs a value");
            opts.cfg.mode = if (std.ascii.toUpper(args[i][0]) == 'B') .b else .a;
        } else if (std.mem.eql(u8, a, "--update")) {
            i += 1;
            if (i >= args.len) return badArg("--update needs a value");
            opts.cfg.update = if (std.mem.eql(u8, args[i], "refit")) .refit else .rebuild;
        } else if (std.mem.eql(u8, a, "--n")) {
            i += 1;
            opts.cfg.n = try intArg(args, i);
        } else if (std.mem.eql(u8, a, "--m")) {
            i += 1;
            opts.cfg.m = try intArg(args, i);
        } else if (std.mem.eql(u8, a, "--res")) {
            i += 1;
            opts.cfg.res = try intArg(args, i);
        } else if (std.mem.eql(u8, a, "--frames")) {
            i += 1;
            opts.cfg.frames = try intArg(args, i);
        } else if (std.mem.eql(u8, a, "--warmup")) {
            i += 1;
            opts.cfg.warmup = try intArg(args, i);
        } else if (std.mem.eql(u8, a, "--spacing")) {
            i += 1;
            if (i >= args.len) return badArg("--spacing needs a value");
            opts.cfg.spacing = std.fmt.parseFloat(f32, args[i]) catch return badArg("--spacing: not a number");
        } else if (std.mem.eql(u8, a, "--csv")) {
            i += 1;
            if (i >= args.len) return badArg("--csv needs a path");
            opts.csv_path = args[i];
        } else if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            std.debug.print("{s}", .{usage});
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n\n{s}", .{ a, usage });
            return error.BadArgument;
        }
    }
    if (!opts.matrix and !opts.verify) opts.single = true;
    return opts;
}

fn badArg(comptime msg: []const u8) error{BadArgument} {
    std.debug.print("{s}\n", .{msg});
    return error.BadArgument;
}

fn intArg(args: []const []const u8, i: usize) !u32 {
    if (i >= args.len) return badArg("missing numeric value");
    return std.fmt.parseInt(u32, args[i], 10) catch badArg("not an unsigned integer");
}

fn printSummary(cfg: scene.Config, r: *const scene.Measurement) !void {
    std.debug.print(
        \\mode {s} update {s}  N={d} M={d} R={d}  status={s}
        \\  AS update : med {d:.3} ms   p90 {d:.3} ms
        \\  dispatch  : med {d:.3} ms   p90 {d:.3} ms
        \\  wall      : med {d:.3} ms   (CPU jitter, outside the submit window: {d:.3} ms)
        \\  end-to-end: {d:.3} ms  -> {d:.1} env-images/s
        \\  memory    : TLAS {d} B, BLAS {d} B, instances {d} B
        \\
    , .{
        if (cfg.mode == .a) "A" else "B",
        if (cfg.update == .refit) "refit" else "rebuild",
        cfg.n,
        cfg.m,
        cfg.res,
        r.status.text(),
        r.as_ms_med,
        r.as_ms_p90,
        r.trace_ms_med,
        r.trace_ms_p90,
        r.wall_ms_med,
        r.cpu_ms_med,
        r.as_ms_med + r.trace_ms_med,
        r.fps,
        r.tlas_bytes,
        r.blas_bytes,
        r.instance_bytes,
    });
}

fn emitCsvRow(io: std.Io, path: ?[]const u8, cfg: scene.Config, r: *const scene.Measurement, header: bool) !void {
    var line_buf: [512]u8 = undefined;
    var fbs = std.Io.Writer.fixed(&line_buf);
    if (header) try fbs.writeAll(csv_header);
    try fbs.print("{s},{s},{d},{d},{d},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d},{d},{d},{d:.1},{s}\n", .{
        if (cfg.mode == .a) "A" else "B",
        if (cfg.update == .refit) "refit" else "rebuild",
        cfg.n,
        cfg.m,
        cfg.res,
        r.as_ms_med,
        r.as_ms_p90,
        r.trace_ms_med,
        r.trace_ms_p90,
        r.wall_ms_med,
        r.cpu_ms_med,
        r.tlas_bytes,
        r.blas_bytes,
        r.instance_bytes,
        r.fps,
        r.status.text(),
    });
    try writeOut(io, path, fbs.buffered());
}

fn writeOut(io: std.Io, path: ?[]const u8, bytes: []const u8) !void {
    const p = path orelse return;
    if (std.mem.eql(u8, p, "-")) {
        var buf: [4096]u8 = undefined;
        var w = std.Io.File.stdout().writer(io, &buf);
        try w.interface.writeAll(bytes);
        try w.flush();
        return;
    }
    // Append so individual single-config runs accumulate into one file.
    const f = std.Io.Dir.cwd().createFile(io, p, .{ .truncate = false }) catch |e| switch (e) {
        error.FileNotFound => try std.Io.Dir.cwd().createFile(io, p, .{ .truncate = true }),
        else => return e,
    };
    defer f.close(io);
    const end = try f.length(io);
    try f.writePositionalAll(io, bytes, end);
}

// ---------------------------------------------------------------------------
// Correctness guard
// ---------------------------------------------------------------------------

/// Runs N=16, M=8, R=64 in both modes and checks:
///   1. per-pixel agreement (float tolerance at silhouettes is implicit: the
///      two modes run the same shader over the same geometry, so the only
///      permitted difference is the marker row; anything else is a bug);
///   2. the marker in env 0 never appears in another environment's image.
fn runCorrectnessGuard(
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    io: std.Io,
    base: scene.Config,
    blas: *const meshes.BlasSet,
    vb: gpu.Buffer,
    ib: gpu.Buffer,
    mb: gpu.Buffer,
) !bool {
    var cfg = base;
    cfg.n = 16;
    cfg.m = 8;
    cfg.res = 64;
    cfg.warmup = 1;
    cfg.frames = 2;

    std.debug.print("\n== correctness guard: N={d} M={d} R={d} ==\n", .{ cfg.n, cfg.m, cfg.res });

    cfg.mode = .a;
    const a = try scene.run(ctx, gpa, io, cfg, blas, vb, ib, mb, true);
    defer if (a.images) |img| gpa.free(img);
    const b = try scene.run(ctx, gpa, io, cfg, blas, vb, ib, mb, true);
    defer if (b.images) |img| gpa.free(img);

    if (a.images == null or b.images == null) {
        std.debug.print("guard FAILED: no image data\n", .{});
        return false;
    }
    const img_a = a.images.?;
    const img_b = b.images.?;
    const per_env: usize = @intCast(cfg.res * cfg.res);

    // (1) The two modes must agree pixel for pixel. Geometry, camera and shading
    // are identical in both modes, and env 0 carries the marker in both, so any
    // difference is an isolation or indexing bug. The differing-pixel fraction
    // is reported regardless.
    var differing: u64 = 0;
    for (0..cfg.n) |env| {
        const slice_a = img_a[env * per_env ..][0..per_env];
        const slice_b = img_b[env * per_env ..][0..per_env];
        for (slice_a, slice_b) |pa, pb| {
            if (pa != pb) differing += 1;
        }
    }
    const total_pixels: u64 = @as(u64, cfg.n) * per_env;
    const frac = @as(f64, @floatFromInt(differing)) / @as(f64, @floatFromInt(total_pixels));
    std.debug.print("A-vs-B differing pixels: {d}/{d} ({d:.4}%)\n", .{ differing, total_pixels, frac * 100.0 });

    // A small tolerance for genuinely ambiguous pixels (a ray that hits exactly a
    // shared silhouette edge, or a shadow ray that grazes). The contract allows
    // up to 0.5%.
    const guard_ok = frac <= 0.005;

    // (2) Marker isolation, checked in each mode independently.
    const marker: u32 = 0x00FF00FF;
    var leak_a: u64 = 0;
    var leak_b: u64 = 0;
    var seen_in_env0 = false;
    for (1..cfg.n) |env| {
        for (img_a[env * per_env ..][0..per_env]) |p| {
            if (p == marker) leak_a += 1;
        }
        for (img_b[env * per_env ..][0..per_env]) |p| {
            if (p == marker) leak_b += 1;
        }
    }
    for (img_a[0..per_env]) |p| {
        if (p == marker) {
            seen_in_env0 = true;
            break;
        }
    }
    std.debug.print("marker pixels in env 0: {s}\n", .{if (seen_in_env0) "visible" else "ABSENT"});

    std.debug.print("marker leak: mode A {d} px, mode B {d} px (must be 0)\n", .{ leak_a, leak_b });
    for (0..cfg.n) |env| {
        var ca: u32 = 0;
        var cb: u32 = 0;
        for (img_a[env * per_env ..][0..per_env]) |p| {
            if (p == marker) ca += 1;
        }
        for (img_b[env * per_env ..][0..per_env]) |p| {
            if (p == marker) cb += 1;
        }
        if (ca != 0 or cb != 0) std.debug.print("  env {d}: marker px A={d} B={d}\n", .{ env, ca, cb });
    }

    const isolation_ok = leak_a == 0 and leak_b == 0 and seen_in_env0;
    if (!guard_ok) std.debug.print("guard FAILED: differing fraction {d:.4}% > 0.5%\n", .{frac * 100.0});
    if (!isolation_ok) std.debug.print("guard FAILED: environment isolation violated\n", .{});
    if (guard_ok and isolation_ok) std.debug.print("guard PASSED\n", .{});
    return guard_ok and isolation_ok;
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

fn runMatrix(
    ctx: *gpu.Context,
    gpa: std.mem.Allocator,
    io: std.Io,
    csv_path: ?[]const u8,
    blas: *const meshes.BlasSet,
    vb: gpu.Buffer,
    ib: gpu.Buffer,
    mb: gpu.Buffer,
) !void {
    // Truncate the CSV before the sweep and write the header.
    if (csv_path) |p| {
        if (!std.mem.eql(u8, p, "-")) {
            const f = std.Io.Dir.cwd().createFile(io, p, .{ .truncate = true }) catch |e| switch (e) {
                error.FileNotFound => try std.Io.Dir.cwd().createFile(io, p, .{ .truncate = true }),
                else => return e,
            };
            defer f.close(io);
            try f.writePositionalAll(io, csv_header, 0);
        } else {
            try writeOut(io, csv_path, csv_header);
        }
    }

    const ns = [_]u32{ 256, 1024, 4096 };
    const ms = [_]u32{ 8, 32, 128 };
    const rs = [_]u32{ 64, 128, 256 };
    const modes = [_]scene.Mode{ .a, .b };
    const updates = [_]scene.Update{ .rebuild, .refit };

    var total: u32 = 0;
    var ran: u32 = 0;
    var skipped: u32 = 0;

    for (modes) |mode| {
        for (updates) |update| {
            for (ns) |n| {
                for (ms) |m| {
                    for (rs) |r| {
                        total += 1;
                        const cfg = scene.Config{
                            .mode = mode,
                            .update = update,
                            .n = n,
                            .m = m,
                            .res = r,
                            .frames = 60,
                            .warmup = 10,
                            .spacing = 16.0,
                            .validation = false,
                        };
                        var meas = scene.run(ctx, gpa, io, cfg, blas, vb, ib, mb, false) catch |e| blk: {
                            std.debug.print("config {s}/{s} N={d} M={d} R={d} failed: {s}\n", .{
                                if (mode == .a) "A" else "B",
                                if (update == .refit) "refit" else "rebuild",
                                n,
                                m,
                                r,
                                @errorName(e),
                            });
                            var failed = scene.Measurement{};
                            failed.status = .skip_alloc;
                            break :blk failed;
                        };
                        if (meas.status == .ok) ran += 1 else skipped += 1;
                        std.debug.print("[{d}/{d}] {s} {s} N={d} M={d} R={d}: ", .{
                            total,
                            @as(u32, @intCast(modes.len * updates.len * ns.len * ms.len * rs.len)),
                            if (mode == .a) "A" else "B",
                            if (update == .refit) "refit" else "rebuild",
                            n,
                            m,
                            r,
                        });
                        if (meas.status == .ok) {
                            std.debug.print("AS {d:.3} ms, trace {d:.3} ms, e2e {d:.3} ms\n", .{
                                meas.as_ms_med,
                                meas.trace_ms_med,
                                meas.as_ms_med + meas.trace_ms_med,
                            });
                        } else {
                            std.debug.print("{s}\n", .{meas.status.text()});
                        }
                        try emitCsvRow(io, csv_path, cfg, &meas, false);
                    }
                }
            }
        }
    }
    std.debug.print("\nmatrix complete: {d} configs, {d} measured, {d} skipped\n", .{ total, ran, skipped });
}
