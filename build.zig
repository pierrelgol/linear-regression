const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vaxis = b.dependency("vaxis", .{
        .target = target,
        .optimize = optimize,
    });

    const names = [_]struct { name: []const u8, desc: []const u8 }{
        .{
            .name = "train",
            .desc = "Train and save a model",
        },
        .{
            .name = "predict",
            .desc = "Interactively estimate a price",
        },
        .{
            .name = "precision",
            .desc = "Report model precision",
        },
        .{
            .name = "plot",
            .desc = "Show the data and fitted line",
        },
    };

    const test_options = b.addOptions();

    test_options.addOption([]const u8, "feature", "train");

    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "options", .module = test_options.createModule() },
            .{ .name = "vaxis", .module = vaxis.module("vaxis") },
        },
    });

    const tests = b.addTest(.{ .root_module = test_mod });

    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run all unit tests");

    test_step.dependOn(&run_tests.step);

    const check = b.step("check", "Compile every executable");

    for (names) |item| {
        const opts = b.addOptions();

        opts.addOption([]const u8, "feature", item.name);

        const mod = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "options", .module = opts.createModule() },
                .{ .name = "vaxis", .module = vaxis.module("vaxis") },
            },
        });

        const exe = b.addExecutable(.{ .name = item.name, .root_module = mod });

        b.installArtifact(exe);
        check.dependOn(&exe.step);

        const run = b.addRunArtifact(exe);
        run.step.dependOn(b.getInstallStep());

        if (b.args) |args| {
            run.addArgs(args);
        }

        const step = b.step(item.name, item.desc);

        step.dependOn(&run.step);
    }
}
