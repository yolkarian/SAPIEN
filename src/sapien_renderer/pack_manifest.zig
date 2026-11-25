//! Parser and checker of a shader pack's pack.json; the C interface is pack_manifest.h.
//!
//! The manifest is parsed once per pack and kept for the life of the pack cache. Every
//! allocation lives in one arena owned by the manifest, so destroying it frees everything and
//! the strings handed out through the C interface stay valid until then.

const std = @import("std");
const json = std.json;

pub const ValueType = enum(i32) { bool = 0, int = 1, float = 2, @"enum" = 3 };
pub const Kind = enum(i32) { uniform = 0, variant = 1 };

/// SapienPackValue
pub const CValue = extern struct {
    type: i32,
    integer: i64,
    real: f64,
    string: ?[*]const u8,
    string_len: usize,
};

/// SapienPackParameter
pub const CParameter = extern struct {
    name: [*]const u8,
    name_len: usize,
    kind: i32,
    type: i32,
    default_value: CValue,
    has_minimum: i32,
    has_maximum: i32,
    minimum: f64,
    maximum: f64,
    enum_count: usize,
    description: [*]const u8,
    description_len: usize,
};

pub const Value = union(ValueType) {
    bool: bool,
    int: i64,
    float: f64,
    @"enum": []const u8,

    fn toC(value: Value) CValue {
        var result = CValue{ .type = @backingInt(std.meta.activeTag(value)), .integer = 0, .real = 0, .string = null, .string_len = 0 };
        switch (value) {
            .bool => |b| result.integer = @intFromBool(b),
            .int => |i| result.integer = i,
            .float => |f| result.real = f,
            .@"enum" => |s| {
                result.string = s.ptr;
                result.string_len = s.len;
            },
        }
        return result;
    }
};

pub const Parameter = struct {
    name: []const u8,
    kind: Kind,
    type: ValueType,
    default: Value,
    minimum: ?f64 = null,
    maximum: ?f64 = null,
    /// enum parameters: the allowed values
    values: []const []const u8 = &.{},
    /// variant bool, int and float parameters: the definition
    define: []const u8 = "",
    /// variant enum parameters: the definition of each value, "" for none
    enum_defines: []const []const u8 = &.{},
    description: []const u8 = "",
};

pub const PresetEntry = struct { parameter: usize, value: Value };
pub const Preset = struct { name: []const u8, entries: []const PresetEntry };

pub const Manifest = struct {
    arena: std.heap.ArenaAllocator,
    parameters: []const Parameter,
    presets: []const Preset,
    /// names of the post-processing passes the pack runs, in order
    post: []const []const u8,

    pub fn findParameter(self: *const Manifest, name: []const u8) ?usize {
        for (self.parameters, 0..) |parameter, index| {
            if (std.mem.eql(u8, parameter.name, name)) return index;
        }
        return null;
    }

    pub fn findPreset(self: *const Manifest, name: []const u8) ?usize {
        for (self.presets, 0..) |preset, index| {
            if (std.mem.eql(u8, preset.name, name)) return index;
        }
        return null;
    }

    pub fn deinit(self: *Manifest) void {
        // the manifest itself lives in its arena: copy the arena out before freeing it
        const arena = self.arena;
        arena.deinit();
    }
};

/// A failure with its message; the message lives in the caller's buffer.
pub const Failure = error{Invalid};

pub const Diagnostic = struct {
    buffer: []u8,
    len: usize = 0,

    /// Write the message and return the error to propagate.
    pub fn fail(self: *Diagnostic, comptime format: []const u8, args: anytype) Failure {
        self.write(format, args);
        return error.Invalid;
    }

    pub fn write(self: *Diagnostic, comptime format: []const u8, args: anytype) void {
        if (self.buffer.len == 0) return;
        var writer = std.Io.Writer.fixed(self.buffer[0 .. self.buffer.len - 1]);
        writer.print(format, args) catch {}; // a truncated message is still useful
        self.len = writer.end;
        self.buffer[self.len] = 0;
    }

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }
};

const ParseError = Failure || std.mem.Allocator.Error;

pub fn parse(text: []const u8, diagnostic: *Diagnostic) ParseError!*Manifest {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    errdefer arena.deinit();
    const gpa = arena.allocator();

    const root = json.parseFromSliceLeaky(json.Value, gpa, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return diagnostic.fail("pack.json is not valid JSON ({s})", .{@errorName(err)}),
    };
    if (root != .object) return diagnostic.fail("pack.json must be a JSON object", .{});
    try expectFields(root.object, &.{ "parameters", "presets", "post" }, "pack.json", diagnostic);

    var parameters: std.ArrayList(Parameter) = .empty;
    if (root.object.get("parameters")) |declared| {
        if (declared != .object) return diagnostic.fail("pack.json: \"parameters\" must be an object", .{});
        var it = declared.object.iterator();
        while (it.next()) |entry| {
            try parameters.append(gpa, try parseParameter(gpa, entry.key_ptr.*, entry.value_ptr.*, diagnostic));
        }
    }

    const manifest = try gpa.create(Manifest);
    manifest.* = .{ .arena = undefined, .parameters = parameters.items, .presets = &.{}, .post = &.{} };

    var presets: std.ArrayList(Preset) = .empty;
    if (root.object.get("presets")) |declared| {
        if (declared != .object) return diagnostic.fail("pack.json: \"presets\" must be an object", .{});
        var it = declared.object.iterator();
        while (it.next()) |entry| {
            try presets.append(gpa, try parsePreset(gpa, manifest, entry.key_ptr.*, entry.value_ptr.*, diagnostic));
        }
    }
    manifest.presets = presets.items;

    var post: std.ArrayList([]const u8) = .empty;
    if (root.object.get("post")) |declared| {
        if (declared != .array) return diagnostic.fail("pack.json: \"post\" must be an array of pass names", .{});
        for (declared.array.items) |item| {
            if (item != .string) return diagnostic.fail("pack.json: \"post\" must be an array of pass names", .{});
            // a pass name becomes the file name <name>.comp
            if (item.string.len == 0) return diagnostic.fail("pack.json: \"post\" has an empty pass name", .{});
            for (item.string) |c| {
                if (!std.ascii.isAlphanumeric(c) and c != '_') return diagnostic.fail("pack.json: post pass \"{s}\" may only contain letters, digits and '_'", .{item.string});
            }
            try post.append(gpa, item.string);
        }
    }
    manifest.post = post.items;
    // from here on every allocation goes through the copy inside the manifest
    manifest.arena = arena;
    return manifest;
}

fn expectFields(object: json.ObjectMap, allowed: []const []const u8, context: []const u8, diagnostic: *Diagnostic) Failure!void {
    var it = object.iterator();
    outer: while (it.next()) |entry| {
        for (allowed) |name| {
            if (std.mem.eql(u8, name, entry.key_ptr.*)) continue :outer;
        }
        return diagnostic.fail("{s}: unknown field \"{s}\"", .{ context, entry.key_ptr.* });
    }
}

fn parseParameter(gpa: std.mem.Allocator, name: []const u8, declared: json.Value, diagnostic: *Diagnostic) ParseError!Parameter {
    if (name.len == 0) return diagnostic.fail("pack.json: a parameter has an empty name", .{});
    if (declared != .object) return diagnostic.fail("parameter \"{s}\" must be an object", .{name});
    const object = declared.object;
    try expectFields(object, &.{ "kind", "type", "default", "minimum", "maximum", "values", "define", "description" }, name, diagnostic);

    const kind_name = try requiredString(object, "kind", name, diagnostic);
    const kind = std.meta.stringToEnum(Kind, kind_name) orelse
        return diagnostic.fail("parameter \"{s}\": kind must be \"uniform\" or \"variant\", not \"{s}\"", .{ name, kind_name });
    const type_name = try requiredString(object, "type", name, diagnostic);
    const value_type = std.meta.stringToEnum(ValueType, type_name) orelse
        return diagnostic.fail("parameter \"{s}\": type must be \"bool\", \"int\", \"float\" or \"enum\", not \"{s}\"", .{ name, type_name });
    if (kind == .uniform and value_type == .@"enum") {
        return diagnostic.fail("parameter \"{s}\": a uniform parameter cannot be an enum", .{name});
    }

    var parameter = Parameter{ .name = name, .kind = kind, .type = value_type, .default = undefined };
    if (object.get("description")) |description| {
        if (description != .string) return diagnostic.fail("parameter \"{s}\": description must be a string", .{name});
        parameter.description = description.string;
    }

    if (object.get("minimum")) |minimum| parameter.minimum = try boundOf(minimum, name, "minimum", value_type, diagnostic);
    if (object.get("maximum")) |maximum| parameter.maximum = try boundOf(maximum, name, "maximum", value_type, diagnostic);
    if (parameter.minimum != null and parameter.maximum != null and parameter.minimum.? > parameter.maximum.?) {
        return diagnostic.fail("parameter \"{s}\": minimum is above maximum", .{name});
    }

    if (value_type == .@"enum") {
        const values = object.get("values") orelse return diagnostic.fail("enum parameter \"{s}\" needs \"values\"", .{name});
        if (values != .array or values.array.items.len == 0) {
            return diagnostic.fail("parameter \"{s}\": values must be a non-empty array of strings", .{name});
        }
        const list = try gpa.alloc([]const u8, values.array.items.len);
        for (values.array.items, 0..) |value, index| {
            if (value != .string) return diagnostic.fail("parameter \"{s}\": values must be strings", .{name});
            for (list[0..index]) |previous| {
                if (std.mem.eql(u8, previous, value.string)) {
                    return diagnostic.fail("parameter \"{s}\": value \"{s}\" is listed twice", .{ name, value.string });
                }
            }
            list[index] = value.string;
        }
        parameter.values = list;
    } else if (object.get("values") != null) {
        return diagnostic.fail("parameter \"{s}\": only enum parameters take \"values\"", .{name});
    }

    const define = object.get("define");
    if (kind == .uniform) {
        if (define != null) return diagnostic.fail("parameter \"{s}\": a uniform parameter takes no \"define\"", .{name});
    } else if (value_type == .@"enum") {
        const mapping = define orelse return diagnostic.fail("variant parameter \"{s}\" needs \"define\"", .{name});
        if (mapping != .object) {
            return diagnostic.fail("parameter \"{s}\": define of an enum maps each value to a definition", .{name});
        }
        const defines = try gpa.alloc([]const u8, parameter.values.len);
        for (parameter.values, 0..) |value, index| {
            const definition = mapping.object.get(value) orelse
                return diagnostic.fail("parameter \"{s}\": define has no entry for value \"{s}\"", .{ name, value });
            if (definition != .string) return diagnostic.fail("parameter \"{s}\": define of \"{s}\" must be a string", .{ name, value });
            defines[index] = try checkDefine(definition.string, name, true, diagnostic);
        }
        if (mapping.object.count() != parameter.values.len) {
            return diagnostic.fail("parameter \"{s}\": define maps a value that is not in values", .{name});
        }
        parameter.enum_defines = defines;
    } else {
        const definition = define orelse return diagnostic.fail("variant parameter \"{s}\" needs \"define\"", .{name});
        if (definition != .string) return diagnostic.fail("parameter \"{s}\": define must be a string", .{name});
        parameter.define = try checkDefine(definition.string, name, false, diagnostic);
    }

    const default = object.get("default") orelse return diagnostic.fail("parameter \"{s}\" needs \"default\"", .{name});
    parameter.default = try valueOf(&parameter, default, "default", diagnostic);
    return parameter;
}

fn requiredString(object: json.ObjectMap, field: []const u8, name: []const u8, diagnostic: *Diagnostic) Failure![]const u8 {
    const value = object.get(field) orelse return diagnostic.fail("parameter \"{s}\" needs \"{s}\"", .{ name, field });
    if (value != .string) return diagnostic.fail("parameter \"{s}\": {s} must be a string", .{ name, field });
    return value.string;
}

fn boundOf(value: json.Value, name: []const u8, field: []const u8, value_type: ValueType, diagnostic: *Diagnostic) Failure!f64 {
    if (value_type != .int and value_type != .float) {
        return diagnostic.fail("parameter \"{s}\": only int and float parameters take \"{s}\"", .{ name, field });
    }
    switch (value) {
        .integer => |i| return @floatFromInt(i),
        .float => |f| {
            if (value_type == .int) {
                return diagnostic.fail("parameter \"{s}\": {s} of an int parameter must be an integer", .{ name, field });
            }
            return f;
        },
        else => return diagnostic.fail("parameter \"{s}\": {s} must be a number", .{ name, field }),
    }
}

/// A C preprocessor identifier; "" (no definition) only where allowed.
fn checkDefine(definition: []const u8, name: []const u8, allow_empty: bool, diagnostic: *Diagnostic) Failure![]const u8 {
    if (definition.len == 0) {
        if (allow_empty) return definition;
        return diagnostic.fail("parameter \"{s}\": define must not be empty", .{name});
    }
    for (definition, 0..) |c, index| {
        const ok = std.ascii.isAlphabetic(c) or c == '_' or (index > 0 and std.ascii.isDigit(c));
        if (!ok) return diagnostic.fail("parameter \"{s}\": \"{s}\" is not a preprocessor identifier", .{ name, definition });
    }
    return definition;
}

/// The JSON `value` as a checked value of `parameter`; `context` names it in errors.
fn valueOf(parameter: *const Parameter, value: json.Value, context: []const u8, diagnostic: *Diagnostic) Failure!Value {
    const converted: ?Value = switch (parameter.type) {
        .bool => if (value == .bool) Value{ .bool = value.bool } else null,
        .int => if (value == .integer) Value{ .int = value.integer } else null,
        .float => switch (value) {
            .integer => |i| Value{ .float = @floatFromInt(i) },
            .float => |f| Value{ .float = f },
            else => null,
        },
        .@"enum" => if (value == .string) Value{ .@"enum" = value.string } else null,
    };
    var checked = converted orelse
        return diagnostic.fail("parameter \"{s}\": {s} must be {s}", .{ parameter.name, context, typeDescription(parameter.type) });
    try check(parameter, &checked, diagnostic);
    return checked;
}

fn typeDescription(value_type: ValueType) []const u8 {
    return switch (value_type) {
        .bool => "a bool",
        .int => "an int",
        .float => "a float",
        .@"enum" => "a string",
    };
}

/// Check the type and range of `value` for `parameter`; an int becomes a float for a float
/// parameter.
pub fn check(parameter: *const Parameter, value: *Value, diagnostic: *Diagnostic) Failure!void {
    if (parameter.type == .float and value.* == .int) {
        const integer = value.int; // read before the union changes its active field
        value.* = .{ .float = @floatFromInt(integer) };
    }
    if (std.meta.activeTag(value.*) != parameter.type) {
        return diagnostic.fail("parameter \"{s}\" takes {s}, not {s}", .{ parameter.name, typeDescription(parameter.type), typeDescription(std.meta.activeTag(value.*)) });
    }
    const number: ?f64 = switch (value.*) {
        .int => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
    if (number) |n| {
        if (std.math.isNan(n)) return diagnostic.fail("parameter \"{s}\" cannot be NaN", .{parameter.name});
        if (parameter.minimum) |minimum| {
            if (n < minimum) return diagnostic.fail("parameter \"{s}\" is {d}, below its minimum {d}", .{ parameter.name, n, minimum });
        }
        if (parameter.maximum) |maximum| {
            if (n > maximum) return diagnostic.fail("parameter \"{s}\" is {d}, above its maximum {d}", .{ parameter.name, n, maximum });
        }
    }
    if (value.* == .@"enum") {
        _ = enumIndex(parameter, value.@"enum") orelse
            return diagnostic.fail("parameter \"{s}\" has no value \"{s}\"", .{ parameter.name, value.@"enum" });
    }
}

fn enumIndex(parameter: *const Parameter, value: []const u8) ?usize {
    for (parameter.values, 0..) |candidate, index| {
        if (std.mem.eql(u8, candidate, value)) return index;
    }
    return null;
}

/// The definition a checked value of a variant parameter selects, "" for none.
pub fn defineOf(parameter: *const Parameter, value: Value) []const u8 {
    std.debug.assert(parameter.kind == .variant);
    return switch (value) {
        .bool => |b| if (b) parameter.define else "",
        .int, .float => parameter.define,
        .@"enum" => |s| parameter.enum_defines[enumIndex(parameter, s).?],
    };
}

fn parsePreset(gpa: std.mem.Allocator, manifest: *const Manifest, name: []const u8, declared: json.Value, diagnostic: *Diagnostic) ParseError!Preset {
    if (name.len == 0) return diagnostic.fail("pack.json: a preset has an empty name", .{});
    if (declared != .object) return diagnostic.fail("preset \"{s}\" must be an object", .{name});
    const entries = try gpa.alloc(PresetEntry, declared.object.count());
    var it = declared.object.iterator();
    var index: usize = 0;
    while (it.next()) |entry| : (index += 1) {
        const parameter = manifest.findParameter(entry.key_ptr.*) orelse
            return diagnostic.fail("preset \"{s}\" sets unknown parameter \"{s}\"", .{ name, entry.key_ptr.* });
        const context = try std.fmt.allocPrint(gpa, "the value in preset \"{s}\"", .{name});
        entries[index] = .{ .parameter = parameter, .value = try valueOf(&manifest.parameters[parameter], entry.value_ptr.*, context, diagnostic) };
    }
    return .{ .name = name, .entries = entries };
}

// ---------------------------------------------------------------------------------------
// C interface (pack_manifest.h)

fn manifestOf(handle: *const anyopaque) *const Manifest {
    return @ptrCast(@alignCast(handle));
}

fn diagnosticOf(error_buffer: ?[*]u8, error_size: usize) Diagnostic {
    return .{ .buffer = if (error_buffer) |b| b[0..error_size] else &.{} };
}

fn valueFromC(value: *const CValue) ?Value {
    return switch (value.type) {
        @backingInt(ValueType.bool) => .{ .bool = value.integer != 0 },
        @backingInt(ValueType.int) => .{ .int = value.integer },
        @backingInt(ValueType.float) => .{ .float = value.real },
        @backingInt(ValueType.@"enum") => .{ .@"enum" = if (value.string) |s| s[0..value.string_len] else "" },
        else => null,
    };
}

export fn sapien_pack_manifest_parse(text: [*]const u8, text_len: usize, out: *?*anyopaque, error_buffer: ?[*]u8, error_size: usize) c_int {
    var diagnostic = diagnosticOf(error_buffer, error_size);
    out.* = null;
    const manifest = parse(text[0..text_len], &diagnostic) catch |err| switch (err) {
        error.Invalid => return 1,
        error.OutOfMemory => {
            diagnostic.write("out of memory while parsing pack.json", .{});
            return 1;
        },
    };
    out.* = manifest;
    return 0;
}

export fn sapien_pack_manifest_destroy(handle: ?*anyopaque) void {
    const manifest: *Manifest = @ptrCast(@alignCast(handle orelse return));
    manifest.deinit();
}

export fn sapien_pack_manifest_parameter_count(handle: *const anyopaque) usize {
    return manifestOf(handle).parameters.len;
}

export fn sapien_pack_manifest_parameter(handle: *const anyopaque, index: usize, out: *CParameter) void {
    const parameter = &manifestOf(handle).parameters[index];
    out.* = .{
        .name = parameter.name.ptr,
        .name_len = parameter.name.len,
        .kind = @backingInt(parameter.kind),
        .type = @backingInt(parameter.type),
        .default_value = parameter.default.toC(),
        .has_minimum = @intFromBool(parameter.minimum != null),
        .has_maximum = @intFromBool(parameter.maximum != null),
        .minimum = parameter.minimum orelse 0,
        .maximum = parameter.maximum orelse 0,
        .enum_count = parameter.values.len,
        .description = parameter.description.ptr,
        .description_len = parameter.description.len,
    };
}

export fn sapien_pack_manifest_enum_value(handle: *const anyopaque, parameter: usize, index: usize, value: *[*]const u8, value_len: *usize) void {
    const text = manifestOf(handle).parameters[parameter].values[index];
    value.* = text.ptr;
    value_len.* = text.len;
}

export fn sapien_pack_manifest_find_parameter(handle: *const anyopaque, name: [*]const u8, name_len: usize) i64 {
    const index = manifestOf(handle).findParameter(name[0..name_len]) orelse return -1;
    return @intCast(index);
}

export fn sapien_pack_manifest_check(handle: *const anyopaque, parameter: usize, value: *CValue, error_buffer: ?[*]u8, error_size: usize) c_int {
    var diagnostic = diagnosticOf(error_buffer, error_size);
    const declared = &manifestOf(handle).parameters[parameter];
    var converted = valueFromC(value) orelse {
        diagnostic.write("parameter \"{s}\": unknown value type {d}", .{ declared.name, value.type });
        return 1;
    };
    check(declared, &converted, &diagnostic) catch return 1;
    // enum strings stay the caller's; only the type and number may change
    const string = value.string;
    const string_len = value.string_len;
    value.* = converted.toC();
    if (converted == .@"enum") {
        value.string = string;
        value.string_len = string_len;
    }
    return 0;
}

export fn sapien_pack_manifest_define(handle: *const anyopaque, parameter: usize, value: *const CValue, define: *[*]const u8, define_len: *usize, error_buffer: ?[*]u8, error_size: usize) c_int {
    var diagnostic = diagnosticOf(error_buffer, error_size);
    const declared = &manifestOf(handle).parameters[parameter];
    if (declared.kind != .variant) {
        diagnostic.write("parameter \"{s}\" is a uniform parameter and selects no definition", .{declared.name});
        return 1;
    }
    var converted = valueFromC(value) orelse {
        diagnostic.write("parameter \"{s}\": unknown value type {d}", .{ declared.name, value.type });
        return 1;
    };
    check(declared, &converted, &diagnostic) catch return 1;
    const definition = defineOf(declared, converted);
    define.* = definition.ptr;
    define_len.* = definition.len;
    return 0;
}

export fn sapien_pack_manifest_preset_count(handle: *const anyopaque) usize {
    return manifestOf(handle).presets.len;
}

export fn sapien_pack_manifest_preset_name(handle: *const anyopaque, preset: usize, name: *[*]const u8, name_len: *usize) void {
    const text = manifestOf(handle).presets[preset].name;
    name.* = text.ptr;
    name_len.* = text.len;
}

export fn sapien_pack_manifest_find_preset(handle: *const anyopaque, name: [*]const u8, name_len: usize) i64 {
    const index = manifestOf(handle).findPreset(name[0..name_len]) orelse return -1;
    return @intCast(index);
}

export fn sapien_pack_manifest_preset_entry_count(handle: *const anyopaque, preset: usize) usize {
    return manifestOf(handle).presets[preset].entries.len;
}

export fn sapien_pack_manifest_preset_entry(handle: *const anyopaque, preset: usize, index: usize, parameter: *usize, value: *CValue) void {
    const entry = manifestOf(handle).presets[preset].entries[index];
    parameter.* = entry.parameter;
    value.* = entry.value.toC();
}

export fn sapien_pack_manifest_post_count(handle: *const anyopaque) usize {
    return manifestOf(handle).post.len;
}

export fn sapien_pack_manifest_post_name(handle: *const anyopaque, index: usize, name: *[*]const u8, name_len: *usize) void {
    const text = manifestOf(handle).post[index];
    name.* = text.ptr;
    name_len.* = text.len;
}

// ---------------------------------------------------------------------------------------
// tests: zig test src/sapien_renderer/pack_manifest.zig

const example =
    \\{
    \\  "parameters": {
    \\    "spp": {"kind": "uniform", "type": "int", "default": 32, "minimum": 1, "maximum": 4096},
    \\    "exposure": {"kind": "uniform", "type": "float", "default": 1, "minimum": 0},
    \\    "russianRoulette": {"kind": "uniform", "type": "bool", "default": false},
    \\    "shadows": {"kind": "variant", "type": "bool", "default": false, "define": "SHADOWS"},
    \\    "maxShadowLights": {"kind": "variant", "type": "int", "default": 4, "minimum": 1, "define": "MAX_SHADOW_LIGHTS"},
    \\    "shading": {"kind": "variant", "type": "enum", "values": ["visibility", "lambert", "pbr"],
    \\                "default": "lambert",
    \\                "define": {"visibility": "SHADING_VISIBILITY", "lambert": "", "pbr": "SHADING_PBR"},
    \\                "description": "shading tier"}
    \\  },
    \\  "presets": {"pbr": {"shading": "pbr", "shadows": true}, "fast": {"spp": 1}}
    \\}
;

fn parseForTest(text: []const u8, buffer: []u8) ParseError!*Manifest {
    var diagnostic = Diagnostic{ .buffer = buffer };
    return parse(text, &diagnostic);
}

fn expectParseError(text: []const u8, expected: []const u8) !void {
    var buffer: [256]u8 = undefined;
    var diagnostic = Diagnostic{ .buffer = &buffer };
    try std.testing.expectError(error.Invalid, parse(text, &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message(), expected) == null) {
        std.debug.print("message \"{s}\" lacks \"{s}\"\n", .{ diagnostic.message(), expected });
        return error.TestUnexpectedResult;
    }
}

test "parameters keep file order, types and defaults" {
    var buffer: [256]u8 = undefined;
    const manifest = try parseForTest(example, &buffer);
    defer manifest.deinit();
    try std.testing.expectEqual(@as(usize, 6), manifest.parameters.len);
    try std.testing.expectEqualStrings("spp", manifest.parameters[0].name);
    try std.testing.expectEqual(Value{ .int = 32 }, manifest.parameters[0].default);
    // an integer default of a float parameter becomes a float
    try std.testing.expectEqual(Value{ .float = 1.0 }, manifest.parameters[1].default);
    try std.testing.expectEqual(Kind.variant, manifest.parameters[5].kind);
    try std.testing.expectEqualStrings("shading tier", manifest.parameters[5].description);
}

test "check converts ints for float parameters and enforces ranges and enum values" {
    var buffer: [256]u8 = undefined;
    const manifest = try parseForTest(example, &buffer);
    defer manifest.deinit();
    var diagnostic = Diagnostic{ .buffer = &buffer };

    var exposure = Value{ .int = 2 };
    try check(&manifest.parameters[1], &exposure, &diagnostic);
    try std.testing.expectEqual(Value{ .float = 2.0 }, exposure);

    var spp = Value{ .int = 0 };
    try std.testing.expectError(error.Invalid, check(&manifest.parameters[0], &spp, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message(), "below its minimum 1") != null);

    var wrong = Value{ .float = 2.5 };
    try std.testing.expectError(error.Invalid, check(&manifest.parameters[0], &wrong, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message(), "takes an int, not a float") != null);

    var shading = Value{ .@"enum" = "toon" };
    try std.testing.expectError(error.Invalid, check(&manifest.parameters[5], &shading, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message(), "no value \"toon\"") != null);
}

test "variant parameters select their definitions" {
    var buffer: [256]u8 = undefined;
    const manifest = try parseForTest(example, &buffer);
    defer manifest.deinit();
    try std.testing.expectEqualStrings("SHADOWS", defineOf(&manifest.parameters[3], .{ .bool = true }));
    try std.testing.expectEqualStrings("", defineOf(&manifest.parameters[3], .{ .bool = false }));
    try std.testing.expectEqualStrings("MAX_SHADOW_LIGHTS", defineOf(&manifest.parameters[4], .{ .int = 2 }));
    try std.testing.expectEqualStrings("SHADING_PBR", defineOf(&manifest.parameters[5], .{ .@"enum" = "pbr" }));
    try std.testing.expectEqualStrings("", defineOf(&manifest.parameters[5], .{ .@"enum" = "lambert" }));
}

test "presets expand to checked entries" {
    var buffer: [256]u8 = undefined;
    const manifest = try parseForTest(example, &buffer);
    defer manifest.deinit();
    const pbr = manifest.presets[manifest.findPreset("pbr").?];
    try std.testing.expectEqual(@as(usize, 2), pbr.entries.len);
    try std.testing.expectEqual(@as(usize, 5), pbr.entries[0].parameter);
    try std.testing.expectEqual(@as(?usize, null), manifest.findPreset("missing"));
}

test "post lists pass names in order and rejects path-like names" {
    var buffer: [256]u8 = undefined;
    const manifest = try parseForTest("{\"post\": [\"bloom\", \"tonemap\"]}", &buffer);
    defer manifest.deinit();
    try std.testing.expectEqual(@as(usize, 2), manifest.post.len);
    try std.testing.expectEqualStrings("bloom", manifest.post[0]);
    try std.testing.expectEqualStrings("tonemap", manifest.post[1]);

    const none = try parseForTest(example, &buffer);
    defer none.deinit();
    try std.testing.expectEqual(@as(usize, 0), none.post.len);

    try expectParseError("{\"post\": \"tonemap\"}", "must be an array of pass names");
    try expectParseError("{\"post\": [1]}", "must be an array of pass names");
    try expectParseError("{\"post\": [\"../x\"]}", "may only contain letters, digits and '_'");
    try expectParseError("{\"post\": [\"\"]}", "empty pass name");
}

test "schema errors name the offending field" {
    try expectParseError("[]", "must be a JSON object");
    try expectParseError("{\"parameter\": {}}", "unknown field \"parameter\"");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"uniform\", \"type\": \"int\"}}}", "\"a\" needs \"default\"");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"uniform\", \"type\": \"enum\", \"values\": [\"x\"], \"default\": \"x\"}}}", "cannot be an enum");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"variant\", \"type\": \"bool\", \"default\": true}}}", "needs \"define\"");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"variant\", \"type\": \"bool\", \"default\": true, \"define\": \"1A\"}}}", "not a preprocessor identifier");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"uniform\", \"type\": \"int\", \"default\": 0, \"minimum\": 1}}}", "below its minimum");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"uniform\", \"type\": \"int\", \"default\": 1, \"colour\": 1}}}", "a: unknown field \"colour\"");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"variant\", \"type\": \"enum\", \"values\": [\"x\", \"y\"], \"default\": \"x\", \"define\": {\"x\": \"X\"}}}}", "no entry for value \"y\"");
    try expectParseError("{\"presets\": {\"p\": {\"missing\": 1}}}", "unknown parameter \"missing\"");
    try expectParseError("{\"parameters\": {\"a\": {\"kind\": \"uniform\", \"type\": \"int\", \"default\": 1}}, \"presets\": {\"p\": {\"a\": 1.5}}}", "in preset \"p\" must be an int");
    try expectParseError("{", "not valid JSON");
}

test "C interface round trip" {
    var handle: ?*anyopaque = null;
    var buffer: [256]u8 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), sapien_pack_manifest_parse(example.ptr, example.len, &handle, &buffer, buffer.len));
    defer sapien_pack_manifest_destroy(handle);

    const index = sapien_pack_manifest_find_parameter(handle.?, "exposure", 8);
    try std.testing.expectEqual(@as(i64, 1), index);
    var value = CValue{ .type = @backingInt(ValueType.int), .integer = 3, .real = 0, .string = null, .string_len = 0 };
    try std.testing.expectEqual(@as(c_int, 0), sapien_pack_manifest_check(handle.?, 1, &value, &buffer, buffer.len));
    try std.testing.expectEqual(@backingInt(ValueType.float), value.type);
    try std.testing.expectEqual(@as(f64, 3.0), value.real);

    var define: [*]const u8 = undefined;
    var define_len: usize = 0;
    const pbr = CValue{ .type = @backingInt(ValueType.@"enum"), .integer = 0, .real = 0, .string = "pbr", .string_len = 3 };
    try std.testing.expectEqual(@as(c_int, 0), sapien_pack_manifest_define(handle.?, 5, &pbr, &define, &define_len, &buffer, buffer.len));
    try std.testing.expectEqualStrings("SHADING_PBR", define[0..define_len]);
    try std.testing.expectEqual(@as(c_int, 1), sapien_pack_manifest_define(handle.?, 0, &value, &define, &define_len, &buffer, buffer.len));

    var bad: ?*anyopaque = null;
    try std.testing.expectEqual(@as(c_int, 1), sapien_pack_manifest_parse("{]", 2, &bad, &buffer, buffer.len));
    try std.testing.expectEqual(@as(?*anyopaque, null), bad);
    try std.testing.expect(std.mem.indexOfScalar(u8, &buffer, 0) != null);
}
