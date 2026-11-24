//! CPU construction of an alias table for discrete importance sampling (Vose's algorithm).
//!
//! The C interface is alias_table.h. The renderer builds one table for the environment map
//! texels (weight = luminance x solid angle) and one for emissive triangles (weight = area x
//! emission luminance), uploads it to a GPU buffer, and the shader samples it in O(1) with two
//! random numbers. Allocations use the page allocator so the object links without libc; the
//! caller owns and frees every returned buffer with sapien_alias_table_free.

const std = @import("std");

/// One alias table entry: bin i keeps its own index with probability threshold / 2^32 and
/// otherwise yields `alias`. The shader compares a uniform 32-bit random word against
/// `threshold`; the integer form avoids float rounding of the acceptance probability.
pub const CEntry = extern struct {
    alias: u32,
    threshold: u32,
};

/// The table handed to C: `count` entries and `total`, the sum of the input weights.
pub const CTable = extern struct {
    entries: [*]CEntry,
    count: usize,
    total: f32,
};

const Entry = struct { alias: usize, prob: f64 };

/// Build a table from `weights`. Negative and non-finite weights are treated as zero. A table
/// with no positive weight has `total == 0` and every entry selects itself with probability 0;
/// the shader must never sample it (the renderer sets the light count to zero in that case).
pub fn build(allocator: std.mem.Allocator, weights: []const f32) !Table {
    const count = weights.len;
    var entries = try allocator.alloc(Entry, count);
    errdefer allocator.free(entries);
    for (entries) |*entry| entry.* = .{ .alias = 0, .prob = 0 };

    var total: f64 = 0;
    for (weights) |w| {
        if (std.math.isFinite(w) and w > 0) total += @as(f64, w);
    }
    if (count == 0 or total <= 0) {
        return .{ .entries = entries, .total = 0 };
    }

    const average = total / @as(f64, @floatFromInt(count));
    var scaled = try allocator.alloc(f64, count);
    defer allocator.free(scaled);
    for (weights, 0..) |w, i| {
        scaled[i] = if (std.math.isFinite(w) and w > 0) @as(f64, w) else 0;
    }

    // Vose: partition into "small" (below average) and "large" (at or above) stacks.
    var small = try allocator.alloc(usize, count);
    defer allocator.free(small);
    var large = try allocator.alloc(usize, count);
    defer allocator.free(large);
    var small_len: usize = 0;
    var large_len: usize = 0;
    for (scaled, 0..) |s, i| {
        if (s < average) {
            small[small_len] = i;
            small_len += 1;
        } else {
            large[large_len] = i;
            large_len += 1;
        }
    }

    while (small_len > 0 and large_len > 0) {
        small_len -= 1;
        large_len -= 1;
        const l = small[small_len];
        const g = large[large_len];
        // Fill the small bin completely with `l` plus a slice of `g`.
        entries[l] = .{ .alias = g, .prob = scaled[l] / average };
        scaled[g] = (scaled[g] + scaled[l]) - average;
        if (scaled[g] < average) {
            small[small_len] = g;
            small_len += 1;
        } else {
            large[large_len] = g;
            large_len += 1;
        }
    }
    // Whatever remains is exactly at the average (up to floating point rounding).
    while (large_len > 0) {
        large_len -= 1;
        entries[large[large_len]].prob = 1;
    }
    while (small_len > 0) {
        small_len -= 1;
        entries[small[small_len]].prob = 1;
    }

    return .{ .entries = entries, .total = @floatCast(total) };
}

pub const Table = struct {
    entries: []Entry,
    total: f32,

    pub fn deinit(self: *Table, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
        self.* = undefined;
    }

    /// Convert the f64 probabilities to the integer thresholds the shader reads.
    pub fn toC(self: *const Table, allocator: std.mem.Allocator) !CTable {
        const n = self.entries.len;
        const out = try allocator.alloc(CEntry, n);
        for (self.entries, 0..) |entry, i| {
            const p = std.math.clamp(entry.prob, 0.0, 1.0);
            // scale by 2^32 and keep the result below 2^32 (p == 1 must not wrap to 0)
            const scaled: u64 = @intFromFloat(@round(p * 4294967296.0));
            out[i] = .{
                .alias = @intCast(entry.alias),
                .threshold = @intCast(@min(scaled, 0xffffffff)),
            };
        }
        return .{ .entries = out.ptr, .count = n, .total = self.total };
    }
};

// ---------------------------------------------------------------------------------------
// C interface (alias_table.h)

const c_allocator = std.heap.page_allocator;

export fn sapien_alias_table_build(weights: ?[*]const f32, count: usize, out: *CTable, error_buffer: ?[*]u8, error_size: usize) c_int {
    out.* = .{ .entries = undefined, .count = 0, .total = 0 };
    const slice: []const f32 = if (count == 0) &.{} else (weights orelse {
        writeError(error_buffer, error_size, "alias table: null weights with nonzero count");
        return 1;
    })[0..count];
    var table = build(c_allocator, slice) catch {
        writeError(error_buffer, error_size, "alias table: out of memory");
        return 1;
    };
    defer table.deinit(c_allocator);
    const result = table.toC(c_allocator) catch {
        writeError(error_buffer, error_size, "alias table: out of memory");
        return 1;
    };
    out.* = result;
    return 0;
}

export fn sapien_alias_table_free(table: *CTable) void {
    if (table.count == 0) return;
    const slice = table.entries[0..table.count];
    c_allocator.free(slice);
    table.* = .{ .entries = undefined, .count = 0, .total = 0 };
}

fn writeError(buffer: ?[*]u8, size: usize, message: []const u8) void {
    const b = buffer orelse return;
    if (size == 0) return;
    const n = @min(message.len, size - 1);
    @memcpy(b[0..n], message[0..n]);
    b[n] = 0;
}

// ---------------------------------------------------------------------------------------
// tests: zig test src/sapien_renderer/alias_table.zig

const test_allocator = std.testing.allocator;

fn histogram(weights: []const f32, samples: usize) ![]f64 {
    var table = try build(test_allocator, weights);
    defer table.deinit(test_allocator);
    var c = try table.toC(test_allocator);
    defer test_allocator.free(c.entries[0..c.count]);

    const hist = try test_allocator.alloc(f64, weights.len);
    @memset(hist, 0);
    // A deterministic LCG; the point is the distribution, not the randomness quality.
    var state: u64 = 0x12345678;
    for (0..samples) |_| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const u: f64 = @as(f64, @floatFromInt(state >> 11)) / 9007199254740992.0;
        const index = @as(usize, @intFromFloat(u * @as(f64, @floatFromInt(weights.len)))) % weights.len;
        const entry = c.entries[index];
        state = state *% 6364136223846793005 +% 1442695040888963407;
        const v = state >> 32;
        const selected = if (v < entry.threshold) index else entry.alias;
        hist[selected] += 1;
    }
    return hist;
}

test "uniform weights produce a uniform table" {
    const weights = [_]f32{ 1, 1, 1, 1 };
    var table = try build(test_allocator, &weights);
    defer table.deinit(test_allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 4), table.total, 1e-6);
    for (table.entries) |entry| {
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), entry.prob, 1e-9);
    }
    var c = try table.toC(test_allocator);
    defer test_allocator.free(c.entries[0..c.count]);
    for (c.entries[0..c.count]) |entry| {
        try std.testing.expectEqual(@as(u32, 0xffffffff), entry.threshold);
    }
}

test "zero weights are never selected" {
    const weights = [_]f32{ 0, 0, 0, 0 };
    var table = try build(test_allocator, &weights);
    defer table.deinit(test_allocator);
    try std.testing.expectEqual(@as(f32, 0), table.total);

    const mixed = [_]f32{ 0, 0, 2, 0 };
    var t2 = try build(test_allocator, &mixed);
    defer t2.deinit(test_allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 2), t2.total, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0), t2.entries[0].prob, 1e-9);
    try std.testing.expectEqual(@as(usize, 2), t2.entries[0].alias);
    try std.testing.expectEqual(@as(usize, 2), t2.entries[1].alias);
    try std.testing.expectEqual(@as(usize, 2), t2.entries[3].alias);
}

test "one heavy entry dominates" {
    const weights = [_]f32{ 1000, 0.001, 0.001, 0.001 };
    var table = try build(test_allocator, &weights);
    defer table.deinit(test_allocator);
    try std.testing.expectApproxEqAbs(@as(f32, 1000.003), table.total, 1e-2);
    // bins 1..3 are tiny and must all alias to the heavy bin 0
    for (1..4) |i| {
        try std.testing.expectEqual(@as(usize, 0), table.entries[i].alias);
        try std.testing.expect(table.entries[i].prob < 1e-3);
    }
    // bin 0 keeps most of its own probability
    try std.testing.expect(table.entries[0].prob > 0.9);
}

test "sampled histogram matches the weights" {
    const weights = [_]f32{ 1, 2, 0, 7 };
    const samples: usize = 200000;
    const hist = try histogram(&weights, samples);
    defer test_allocator.free(hist);
    // the zero-weight bin is never selected
    try std.testing.expectEqual(@as(f64, 0), hist[2]);
    var total: f64 = 0;
    for (weights) |w| total += w;
    for (weights, 0..) |w, i| {
        const expected = @as(f64, w) / total;
        const observed = hist[i] / @as(f64, @floatFromInt(samples));
        try std.testing.expectApproxEqAbs(expected, observed, 0.005);
    }
}

test "C interface builds, frees and rejects a null input" {
    const weights = [_]f32{ 3, 1, 0, 6 };
    var out: CTable = undefined;
    var error_buffer: [128]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), sapien_alias_table_build(&weights, weights.len, &out, &error_buffer, error_buffer.len));
    try std.testing.expectEqual(@as(usize, 4), out.count);
    try std.testing.expectApproxEqAbs(@as(f32, 10), out.total, 1e-6);
    sapien_alias_table_free(&out);
    try std.testing.expectEqual(@as(usize, 0), out.count);

    var bad: CTable = undefined;
    try std.testing.expectEqual(@as(c_int, 1), sapien_alias_table_build(null, 3, &bad, &error_buffer, error_buffer.len));
    try std.testing.expectEqualStrings("alias table: null weights with nonzero count", std.mem.sliceTo(&error_buffer, 0));
}
