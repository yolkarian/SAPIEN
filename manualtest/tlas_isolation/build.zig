const std = @import("std");

const Translator = @import("translate_c").Translator;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // GLSL -> SPIR-V with the prebuilt Khronos glslang. The output is embedded
    // in the binary, so runs need no shader files next to the executable.
    const glslang = b.option([]const u8, "glslang", "path to the glslang binary") orelse "glslang";
    const glsl = b.addSystemCommand(&.{ glslang, "-V", "--target-env", "vulkan1.3", "-S", "comp" });
    glsl.addFileArg(b.path("shaders/trace.comp"));
    glsl.addArg("-o");
    const spv = glsl.addOutputFileArg("trace.spv");

    // Vulkan C API bound through translate-c (Zig 0.17 removed @cImport).
    const tc = b.dependency("translate_c", .{ .target = target, .optimize = .Debug });
    const vulkan_headers = b.option(
        []const u8,
        "vulkan-headers",
        "path to Vulkan-Headers/include",
    ) orelse "/usr/include";

    const wrapper = b.addWriteFiles().add("vulkan_wrap.h",
        \\#include <vulkan/vulkan.h>
        \\
    );
    const vk_mod = Translator.init(tc, .{
        .name = "vk",
        .c_source_file = wrapper,
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    vk_mod.addIncludePath(.{ .cwd_relative = vulkan_headers });

    const exe = b.addExecutable(.{
        .name = "tlas_isolation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("vk", vk_mod.mod);
    // Zig 0.17 no longer resolves @embedFile through --embed-dir, so the SPIR-V
    // is materialised next to the sources it is embedded from. addUpdateSourceFiles
    // runs before compilation and is cached against the glslang output.
    const stage_spv = b.addUpdateSourceFiles();
    stage_spv.addCopyFileToSource(spv, "src/trace.spv");
    exe.step.dependOn(&stage_spv.step);

    // Link the Vulkan loader by absolute path: only the runtime .so.1 is present
    // on the target host, with no development symlink.
    const vulkan_loader = b.option(
        []const u8,
        "vulkan-loader",
        "path to libvulkan.so.1",
    ) orelse "/usr/lib/x86_64-linux-gnu/libvulkan.so.1";
    exe.root_module.addObjectFile(.{ .cwd_relative = vulkan_loader });

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run the benchmark");
    run_step.dependOn(&run_cmd.step);
}
