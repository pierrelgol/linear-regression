const std = @import("std");
const options = @import("options");
const vaxis = @import("vaxis");

pub const LinearRegression = struct {
    pub const Feature = enum {
        train,
        predict,
        precision,
        plot,
    };

    pub const Point = struct {
        km: f64,
        price: f64,
    };

    pub const Model = struct {
        theta0: f64 = 0,
        theta1: f64 = 0,

        pub fn estimate(self: Model, km: f64) f64 {
            return self.theta0 + self.theta1 * km;
        }
    };

    pub const Metrics = struct {
        r2: f64,
        mae: f64,
        rmse: f64,
    };

    pub const Config = struct {
        data: []const u8 = "data/data.csv",
        weights: []const u8 = "weight.pt",
        epochs: usize = 10000,
        learning_rate: f64 = 0.1,
        epsilon: f64 = 1e-10,
        seed: u64 = 0,
        train_split: f64 = 1,
        help: bool = false,
    };

    fn finite(x: f64) bool {
        return !std.math.isNan(x) and !std.math.isInf(x);
    }

    const Option = enum {
        weights,
        data,
        epochs,
        learning_rate,
        epsilon,
        seed,
        train_split,
    };

    fn identifyOption(flag: []const u8, comptime feature: Feature) ?Option {
        if (std.mem.eql(u8, flag, "--weights")) {
            return .weights;
        }

        switch (feature) {
            .predict => {},

            .train, .precision, .plot => {
                if (std.mem.eql(u8, flag, "--data")) {
                    return .data;
                }
            },
        }

        switch (feature) {
            .train => {
                if (std.mem.eql(u8, flag, "--epochs")) {
                    return .epochs;
                }

                if (std.mem.eql(u8, flag, "--learning-rate")) {
                    return .learning_rate;
                }

                if (std.mem.eql(u8, flag, "--epsilon")) {
                    return .epsilon;
                }

                if (std.mem.eql(u8, flag, "--seed")) {
                    return .seed;
                }

                if (std.mem.eql(u8, flag, "--train-split")) {
                    return .train_split;
                }
            },

            else => {},
        }

        return null;
    }

    pub fn parseArgs(it: *std.process.Args.Iterator, comptime feature: Feature) !Config {
        var c: Config = .{};

        while (it.next()) |arg| {
            const is_long_help = std.mem.eql(u8, arg, "--help");

            const is_short_help = std.mem.eql(u8, arg, "-h");

            if (is_long_help or is_short_help) {
                c.help = true;

                return c;
            }

            const equals = std.mem.indexOfScalar(u8, arg, '=');

            const flag =
                if (equals) |i|
                    arg[0..i]
                else
                    arg;

            const option = identifyOption(flag, feature) orelse return error.UnknownArgument;

            const value = if (equals) |i|
                arg[i + 1 ..]
            else
                it.next() orelse return error.MissingArgument;

            if (value.len == 0) {
                return error.MissingArgument;
            }

            switch (option) {
                .weights => c.weights = value,
                .data => c.data = value,
                .epochs => c.epochs = try std.fmt.parseInt(usize, value, 10),
                .learning_rate => c.learning_rate = try std.fmt.parseFloat(f64, value),
                .epsilon => c.epsilon = try std.fmt.parseFloat(f64, value),
                .seed => c.seed = try std.fmt.parseInt(u64, value, 10),
                .train_split => c.train_split = try std.fmt.parseFloat(f64, value),
            }
        }

        if (c.epochs == 0) {
            return error.InvalidArgument;
        }

        if (!finite(c.learning_rate) or c.learning_rate <= 0) {
            return error.InvalidArgument;
        }

        if (!finite(c.epsilon) or c.epsilon < 0) {
            return error.InvalidArgument;
        }

        if (!finite(c.train_split)) {
            return error.InvalidArgument;
        }

        if (c.train_split <= 0 or c.train_split > 1) {
            return error.InvalidArgument;
        }

        return c;
    }

    pub fn parseCsv(gpa: std.mem.Allocator, text: []const u8) ![]Point {
        var rows: std.ArrayList(Point) = .empty;
        errdefer rows.deinit(gpa);

        var lines = std.mem.splitScalar(u8, text, '\n');
        var header = false;

        while (lines.next()) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");

            if (line.len == 0) {
                continue;
            }

            if (!header) {
                if (!std.mem.eql(u8, line, "km,price")) {
                    return error.InvalidHeader;
                }

                header = true;

                continue;
            }

            var fields = std.mem.splitScalar(u8, line, ',');
            const a = fields.next() orelse return error.InvalidRow;
            const b = fields.next() orelse return error.InvalidRow;

            if (fields.next() != null) {
                return error.InvalidRow;
            }

            const km = std.fmt.parseFloat(f64, std.mem.trim(u8, a, " \t")) catch return error.InvalidRow;
            const price = std.fmt.parseFloat(f64, std.mem.trim(u8, b, " \t")) catch return error.InvalidRow;

            if (!finite(km) or !finite(price)) {
                return error.InvalidRow;
            }

            try rows.append(gpa, .{ .km = km, .price = price });
        }

        if (!header or rows.items.len == 0) {
            return error.EmptyDataset;
        }

        return rows.toOwnedSlice(gpa);
    }

    fn loadData(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]Point {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 * 1024 * 1024));
        defer gpa.free(bytes);

        return parseCsv(gpa, bytes);
    }

    pub fn parseWeights(text: []const u8) !Model {
        if (std.mem.trim(u8, text, " \t\r\n").len == 0) {
            return .{};
        }

        var lines = std.mem.splitScalar(u8, text, '\n');
        const header = lines.next() orelse return error.InvalidWeights;
        const trimmed_header = std.mem.trim(u8, header, " \t\r");

        if (!std.mem.eql(u8, trimmed_header, "theta0,theta1")) {
            return error.InvalidWeights;
        }

        const row = std.mem.trim(u8, lines.next() orelse return error.InvalidWeights, " \t\r");
        var f = std.mem.splitScalar(u8, row, ',');
        const a = f.next() orelse return error.InvalidWeights;
        const b = f.next() orelse return error.InvalidWeights;

        if (f.next() != null) {
            return error.InvalidWeights;
        }

        const m: Model = .{
            .theta0 = std.fmt.parseFloat(f64, a) catch return error.InvalidWeights,
            .theta1 = std.fmt.parseFloat(f64, b) catch return error.InvalidWeights,
        };

        if (!finite(m.theta0) or !finite(m.theta1)) {
            return error.InvalidWeights;
        }

        while (lines.next()) |line| {
            if (std.mem.trim(u8, line, " \t\r").len != 0) {
                return error.InvalidWeights;
            }
        }

        return m;
    }

    fn loadWeights(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Model {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1024 * 1024));
        defer gpa.free(bytes);

        return parseWeights(bytes);
    }

    fn shouldLogProgress(completed: usize, epochs: usize, interval: usize, early_interval: usize, converged: bool) bool {
        if (completed == 1) {
            return true;
        }

        if (completed == epochs) {
            return true;
        }

        if (completed % interval == 0) {
            return true;
        }

        if (completed < interval and completed % early_interval == 0) {
            return true;
        }

        if (converged) {
            return true;
        }

        return false;
    }

    pub fn train(points: []const Point, epochs: usize, rate: f64, epsilon: f64) !Model {
        var min_x = points[0].km;
        var max_x = min_x;
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;

        for (points) |p| {
            min_x = @min(min_x, p.km);
            max_x = @max(max_x, p.km);

            sum_x += p.km;
            sum_y += p.price;
        }

        if (min_x == max_x) {
            return error.ConstantMileage;
        }

        const n: f64 = @floatFromInt(points.len);
        const mx = sum_x / n;
        const my = sum_y / n;
        var vx: f64 = 0;
        var vy: f64 = 0;

        for (points) |p| {
            vx += (p.km - mx) * (p.km - mx);
            vy += (p.price - my) * (p.price - my);
        }

        const sx = @sqrt(vx / n);
        const sy = if (vy == 0) 1 else @sqrt(vy / n);

        if (!finite(sx) or sx == 0 or !finite(sy)) {
            return error.UnusableScale;
        }

        var t0: f64 = 0;
        var t1: f64 = 0;
        var epoch: usize = 0;

        while (epoch < epochs) : (epoch += 1) {
            var g0: f64 = 0;
            var g1: f64 = 0;
            var loss: f64 = 0;

            for (points) |p| {
                const x = (p.km - mx) / sx;
                const y = (p.price - my) / sy;
                const e = t0 + t1 * x - y;

                g0 += e;
                g1 += e * x;
                loss += e * e;
            }

            g0 /= n;
            g1 /= n;

            if (!finite(g0) or !finite(g1) or !finite(loss)) {
                return error.TrainingDiverged;
            }

            const completed = epoch + 1;
            const log_interval = @max(@as(usize, 1), epochs / 20);
            const early_log_interval = @max(@as(usize, 1), epochs / 200);
            const converged = @max(@abs(g0), @abs(g1)) <= epsilon;

            if (shouldLogProgress(completed, epochs, log_interval, early_log_interval, converged)) {
                const progress = 100 * @as(f64, @floatFromInt(completed)) / @as(f64, @floatFromInt(epochs));
                std.log.info("progress {d:>6.2}% | epoch {d}/{d} | loss={d:.10}", .{ progress, completed, epochs, loss / (2 * n) });
            }

            if (converged) {
                break;
            }

            t0 -= rate * g0;
            t1 -= rate * g1;

            if (!finite(t0) or !finite(t1)) {
                return error.TrainingDiverged;
            }
        }

        const model: Model = .{
            .theta1 = sy * t1 / sx,
            .theta0 = my + sy * t0 - (sy * t1 / sx) * mx,
        };

        if (!finite(model.theta0) or !finite(model.theta1)) {
            return error.TrainingDiverged;
        }

        return model;
    }

    pub fn metrics(points: []const Point, model: Model) Metrics {
        var mean: f64 = 0;

        for (points) |p| {
            mean += p.price;
        }

        mean /= @floatFromInt(points.len);

        var ae: f64 = 0;
        var se: f64 = 0;
        var total: f64 = 0;

        for (points) |p| {
            const e = model.estimate(p.km) - p.price;

            ae += @abs(e);
            se += e * e;
            total += (p.price - mean) * (p.price - mean);
        }

        const n: f64 = @floatFromInt(points.len);
        const r2: f64 = if (total == 0)
            switch (se <= 1e-20) {
                true => 1,
                false => 0,
            }
        else
            1 - se / total;

        return .{
            .r2 = r2,
            .mae = ae / n,
            .rmse = @sqrt(se / n),
        };
    }

    fn checkedMetrics(points: []const Point, model: Model) !Metrics {
        const result = metrics(points, model);

        if (!finite(result.r2) or !finite(result.mae) or !finite(result.rmse)) {
            return error.MetricOverflow;
        }

        return result;
    }

    fn save(model: Model, gpa: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        if (!finite(model.theta0) or !finite(model.theta1)) {
            return error.InvalidModel;
        }

        const contents = try std.fmt.allocPrint(gpa, "theta0,theta1\n{d},{d}\n", .{ model.theta0, model.theta1 });
        defer gpa.free(contents);

        const nonce = std.Io.Timestamp.now(io, .awake).nanoseconds;
        const tmp = try std.fmt.allocPrint(gpa, "{s}.tmp.{x}", .{ path, @as(u96, @bitCast(nonce)) });
        defer gpa.free(tmp);

        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = contents });
        errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), path, io);
    }

    fn usage(w: *std.Io.Writer, comptime feature: Feature) !void {
        switch (feature) {
            .train => try trainUsage(w),
            .predict => try predictUsage(w),
            .precision => try precisionUsage(w),
            .plot => try plotUsage(w),
        }
    }

    fn commonOptionUsage(w: *std.Io.Writer) !void {
        try w.writeAll(
            \\
            \\Option syntax:
            \\  Both `--option value` and `--option=value` are accepted.
            \\  -h, --help             Show this help and exit.
            \\
        );
    }

    fn trainUsage(w: *std.Io.Writer) !void {
        try w.writeAll(
            \\Usage: train [OPTIONS]
            \\
            \\Train a linear regression model from mileage and price data.
            \\
            \\Options:
            \\  --data PATH            Input CSV file. Default: data/data.csv
            \\  --weights PATH         Output weight file. Default: weight.pt
            \\  --epochs N             Maximum gradient steps. Default: 10000
            \\  --learning-rate X      Positive gradient step size. Default: 0.1
            \\  --epsilon X            Non-negative convergence threshold. Default: 1e-10
            \\  --seed N               Shuffle seed for split training. Default: 0
            \\  --train-split X        Training fraction in (0, 1]. Default: 1.0
            \\
            \\Example:
            \\  train --data data/data.csv --weights weight.pt --epochs 10000
            \\
        );

        try commonOptionUsage(w);
    }

    fn predictUsage(w: *std.Io.Writer) !void {
        try w.writeAll(
            \\Usage: predict [OPTIONS]
            \\
            \\Load a model, prompt for mileage, and print the estimated price.
            \\
            \\Options:
            \\  --weights PATH         Input weight file. Default: weight.pt
            \\
            \\Example:
            \\  predict --weights weight.pt
            \\
        );

        try commonOptionUsage(w);
    }

    fn precisionUsage(w: *std.Io.Writer) !void {
        try w.writeAll(
            \\Usage: precision [OPTIONS]
            \\
            \\Report R2, mean absolute error, and root mean squared error.
            \\
            \\Options:
            \\  --data PATH            Input CSV file. Default: data/data.csv
            \\  --weights PATH         Input weight file. Default: weight.pt
            \\
            \\Example:
            \\  precision --data data/data.csv --weights weight.pt
            \\
        );

        try commonOptionUsage(w);
    }

    fn plotUsage(w: *std.Io.Writer) !void {
        try w.writeAll(
            \\Usage: plot [OPTIONS]
            \\
            \\Open a responsive terminal plot with linear axes.
            \\Price is horizontal (x); mileage in kilometers is vertical (y).
            \\The trained model is green; a closed-form ordinary least-squares
            \\reference model is blue. Axis labels show five precise graduations.
            \\
            \\Options:
            \\  --data PATH            Input CSV file. Default: data/data.csv
            \\  --weights PATH         Input weight file. Default: weight.pt
            \\
            \\Controls:
            \\  q, Escape, Ctrl-C       Close the plot.
            \\
            \\Example:
            \\  plot --data data/data.csv --weights weight.pt
            \\
        );

        try commonOptionUsage(w);
    }

    pub fn run(init: std.process.Init, comptime feature: Feature) !void {
        var args = try init.minimal.args.iterateAllocator(init.gpa);
        defer args.deinit();

        _ = args.next();
        const c = try parseArgs(&args, feature);

        var out_buf: [4096]u8 = undefined;
        var fw: std.Io.File.Writer = .init(.stdout(), init.io, &out_buf);
        const out = &fw.interface;

        if (c.help) {
            try usage(out, feature);
            try out.flush();
            return;
        }

        switch (feature) {
            .train => {
                var points = try loadData(init.gpa, init.io, c.data);

                defer init.gpa.free(points);

                if (c.train_split < 1) {
                    var prng = std.Random.DefaultPrng.init(c.seed);
                    prng.random().shuffle(Point, points);
                    const count = @as(usize, @intFromFloat(@floor(c.train_split * @as(f64, @floatFromInt(points.len)))));

                    if (count == 0 or count >= points.len) {
                        return error.InvalidSplit;
                    }

                    const model = try train(points[0..count], c.epochs, c.learning_rate, c.epsilon);
                    try save(model, init.gpa, init.io, c.weights);
                    const mt = try checkedMetrics(points[0..count], model);
                    const mv = try checkedMetrics(points[count..], model);

                    std.log.info("theta0={d}, theta1={d}; train R2={d:.6}; validation R2={d:.6}", .{ model.theta0, model.theta1, mt.r2, mv.r2 });
                } else {
                    const model = try train(points, c.epochs, c.learning_rate, c.epsilon);
                    try save(model, init.gpa, init.io, c.weights);
                    const m = try checkedMetrics(points, model);
                    std.log.info("theta0={d}, theta1={d}; R2={d:.6}, MAE={d:.2}, RMSE={d:.2}", .{ model.theta0, model.theta1, m.r2, m.mae, m.rmse });
                }
            },

            .predict => {
                const model = try loadWeights(init.gpa, init.io, c.weights);
                try out.writeAll("Enter mileage (km): ");
                try out.flush();
                var in_buf: [256]u8 = undefined;
                var fr: std.Io.File.Reader = .init(.stdin(), init.io, &in_buf);
                const line = try fr.interface.takeDelimiterInclusive('\n');
                const km = try std.fmt.parseFloat(f64, std.mem.trim(u8, line, " \t\r\n"));

                if (!finite(km) or km < 0) {
                    return error.InvalidMileage;
                }

                const estimate = model.estimate(km);

                if (!finite(estimate)) {
                    return error.EstimateOverflow;
                }

                try out.print("Estimated price: {d:.2}\n", .{estimate});
            },

            .precision => {
                const points = try loadData(init.gpa, init.io, c.data);
                defer init.gpa.free(points);

                const m = try checkedMetrics(points, try loadWeights(init.gpa, init.io, c.weights));
                try out.print("R²: {d:.6}\nMAE: {d:.2}\nRMSE: {d:.2}\n", .{ m.r2, m.mae, m.rmse });
            },

            .plot => {
                const points = try loadData(init.gpa, init.io, c.data);
                defer init.gpa.free(points);

                const model = try loadWeights(init.gpa, init.io, c.weights);
                _ = try checkedMetrics(points, model);
                try runPlot(init, points, model);
            },
        }

        try out.flush();
    }

    pub fn plotCoordinate(v: f64, min: f64, max: f64, size: usize) usize {
        if (size <= 1 or max <= min) {
            return 0;
        }

        const scaled = (v - min) / (max - min) * @as(f64, @floatFromInt(size - 1));
        return @min(size - 1, @as(usize, @intFromFloat(@max(0, @min(@as(f64, @floatFromInt(size - 1)), scaled)))));
    }

    pub fn ordinaryLeastSquares(points: []const Point) !Model {
        const count: f64 = @floatFromInt(points.len);
        var sum_x: f64 = 0;
        var sum_y: f64 = 0;

        for (points) |point| {
            sum_x += point.km;
            sum_y += point.price;
        }

        const mean_x = sum_x / count;
        const mean_y = sum_y / count;

        var covariance: f64 = 0;
        var variance_x: f64 = 0;

        for (points) |point| {
            const centered_x = point.km - mean_x;
            covariance += centered_x * (point.price - mean_y);
            variance_x += centered_x * centered_x;
        }

        if (!finite(covariance) or !finite(variance_x) or variance_x == 0) {
            return error.UnusableReferenceModel;
        }

        const theta1 = covariance / variance_x;
        const theta0 = mean_y - theta1 * mean_x;

        if (!finite(theta0) or !finite(theta1)) {
            return error.UnusableReferenceModel;
        }

        return .{
            .theta0 = theta0,
            .theta1 = theta1,
        };
    }

    const PlotWidget = struct {
        points: []const Point,
        model: Model,
        reference: Model,

        fn widget(self: *PlotWidget) vaxis.vxfw.Widget {
            return .{ .userdata = self, .eventHandler = event, .drawFn = draw };
        }

        fn shouldExit(key: vaxis.Key) bool {
            if (key.matches('q', .{})) {
                return true;
            }

            if (key.matches(vaxis.Key.escape, .{})) {
                return true;
            }

            if (key.matches('c', .{ .ctrl = true })) {
                return true;
            }

            return false;
        }

        fn event(_: *anyopaque, ctx: *vaxis.vxfw.EventContext, ev: vaxis.vxfw.Event) anyerror!void {
            switch (ev) {
                .key_press => |key| {
                    if (shouldExit(key)) {
                        ctx.quit = true;

                        ctx.consumeEvent();
                    }
                },

                else => {},
            }
        }

        fn cell(surface: vaxis.vxfw.Surface, col: u16, row: u16, grapheme: []const u8, style: vaxis.Style) void {
            surface.writeCell(col, row, .{ .char = .{ .grapheme = grapheme }, .style = style });
        }

        fn text(surface: vaxis.vxfw.Surface, col: u16, row: u16, value: []const u8, style: vaxis.Style) void {
            if (row >= surface.size.height) {
                return;
            }

            for (value, 0..) |_, i| {
                const x: u16 = col +| @as(u16, @intCast(i));

                if (x >= surface.size.width) {
                    break;
                }

                cell(surface, x, row, value[i .. i + 1], style);
            }
        }

        fn draw(ptr: *anyopaque, ctx: vaxis.vxfw.DrawContext) std.mem.Allocator.Error!vaxis.vxfw.Surface {
            const self: *PlotWidget = @ptrCast(@alignCast(ptr));
            const width: u16 = @max(ctx.min.width, ctx.max.width orelse 80);
            const height: u16 = @max(ctx.min.height, ctx.max.height orelse 24);
            const surface = try vaxis.vxfw.Surface.init(ctx.arena, self.widget(), .{ .width = width, .height = height });
            const cyan: vaxis.Style = .{ .fg = .{ .rgb = .{ 90, 210, 255 } }, .bold = true };
            const yellow: vaxis.Style = .{ .fg = .{ .rgb = .{ 255, 190, 50 } }, .bold = true };
            const green: vaxis.Style = .{ .fg = .{ .rgb = .{ 80, 235, 130 } } };

            text(surface, 1, 0, "Lineare regression price (x) / mileage in km (y)", cyan);

            if (width < 56 or height < 20) {
                text(surface, 1, 2, "Terminal too smal resize to at least 56 x 20", .{ .fg = .{ .rgb = .{ 255, 90, 90 } } });

                return surface;
            }

            const left: u16 = 12;
            const top: u16 = 2;
            const plot_w: u16 = width - left - 2;
            const plot_h: u16 = height - top - 9;

            var minx = self.points[0].price;
            var maxx = minx;
            var miny = self.points[0].km;
            var maxy = miny;

            for (self.points) |p| {
                minx = @min(minx, p.price);
                maxx = @max(maxx, p.price);
                miny = @min(miny, p.km);
                maxy = @max(maxy, p.km);
            }

            minx = @min(minx, @min(self.model.estimate(miny), self.model.estimate(maxy)));
            maxx = @max(maxx, @max(self.model.estimate(miny), self.model.estimate(maxy)));
            minx = @min(minx, @min(self.reference.estimate(miny), self.reference.estimate(maxy)));
            maxx = @max(maxx, @max(self.reference.estimate(miny), self.reference.estimate(maxy)));

            for (0..plot_h) |y| {
                cell(surface, left, top + @as(u16, @intCast(y)), "|", cyan);
            }

            for (0..plot_w) |x| {
                cell(surface, left + @as(u16, @intCast(x)), top + plot_h, "-", cyan);
            }

            const sample_count: usize = @max(plot_w, plot_h) * 4;

            for (0..sample_count) |sample| {
                const fraction = @as(f64, @floatFromInt(sample)) / @as(f64, @floatFromInt(sample_count - 1));
                const km = miny + (maxy - miny) * fraction;
                const x = plotCoordinate(self.reference.estimate(km), minx, maxx, plot_w);
                const y = plotCoordinate(km, miny, maxy, plot_h);

                cell(surface, left + @as(u16, @intCast(x)), top + plot_h - 1 - @as(u16, @intCast(y)), "=", .{ .fg = .{ .rgb = .{ 70, 130, 255 } }, .bold = true });
            }

            for (0..sample_count) |sample| {
                if (sample % 2 != 0) {
                    continue;
                }

                const fraction = @as(f64, @floatFromInt(sample)) / @as(f64, @floatFromInt(sample_count - 1));
                const km = miny + (maxy - miny) * fraction;
                const x = plotCoordinate(self.model.estimate(km), minx, maxx, plot_w);
                const y = plotCoordinate(km, miny, maxy, plot_h);

                cell(surface, left + @as(u16, @intCast(x)), top + plot_h - 1 - @as(u16, @intCast(y)), "-", green);
            }

            for (self.points) |p| {
                const x = plotCoordinate(p.price, minx, maxx, plot_w);
                const y = plotCoordinate(p.km, miny, maxy, plot_h);

                cell(surface, left + @as(u16, @intCast(x)), top + plot_h - 1 - @as(u16, @intCast(y)), "*", yellow);
            }

            const tick_count: usize = 5;

            for (0..tick_count) |tick| {
                const fraction = @as(f64, @floatFromInt(tick)) / @as(f64, tick_count - 1);
                const x = @as(u16, @intFromFloat(@round(fraction * @as(f64, @floatFromInt(plot_w - 1)))));
                const x_value = minx + (maxx - minx) * fraction;
                const x_label = try std.fmt.allocPrint(ctx.arena, "{e:.2}", .{x_value});

                text(surface, left + x -| @as(u16, @intCast(@min(x_label.len / 2, x))), top + plot_h + 1, x_label, .{});

                const y = @as(u16, @intFromFloat(@round(fraction * @as(f64, @floatFromInt(plot_h - 1)))));
                const y_value = miny + (maxy - miny) * fraction;
                const y_label = try std.fmt.allocPrint(ctx.arena, "{e:.2}", .{y_value});
                const y_col = left -| @as(u16, @intCast(y_label.len + 1));

                text(surface, y_col, top + plot_h - 1 - y, y_label, .{});
            }

            const m = metrics(self.points, self.model);
            const reference_metrics = metrics(self.points, self.reference);
            const stats = try std.fmt.allocPrint(ctx.arena, "* data  - trained (green): theta0={d:.6} theta1={d:.10}", .{ self.model.theta0, self.model.theta1 });
            const reference_stats = try std.fmt.allocPrint(ctx.arena, "= OLS truth (blue): theta0={d:.6} theta1={d:.10}", .{ self.reference.theta0, self.reference.theta1 });
            const scores = try std.fmt.allocPrint(ctx.arena, "trained R2={d:.8} MAE={d:.4} RMSE={d:.4} | OLS R2={d:.8} RMSE={d:.4}", .{ m.r2, m.mae, m.rmse, reference_metrics.r2, reference_metrics.rmse });

            text(surface, 1, height - 5, stats, .{});
            text(surface, 1, height - 4, reference_stats, .{ .fg = .{ .rgb = .{ 70, 130, 255 } } });
            text(surface, 1, height - 3, scores, .{});
            text(surface, 1, height - 2, "Axes: linear price (x) and mileage km (y) | q/Esc/Ctrl-C: exit", .{});

            return surface;
        }
    };

    fn runPlot(init: std.process.Init, points: []const Point, model: Model) !void {
        const reference = try ordinaryLeastSquares(points);
        var tty_buffer: [4096]u8 = undefined;

        var app = try vaxis.vxfw.App.init(init.io, init.gpa, init.environ_map, &tty_buffer);
        defer app.deinit();

        var plot: PlotWidget = .{
            .points = points,
            .model = model,
            .reference = reference,
        };

        try app.run(plot.widget(), .{});
    }
};

pub fn main(init: std.process.Init) void {
    LinearRegression.run(init, @field(LinearRegression.Feature, options.feature)) catch |err| {
        std.log.err("{s} failed: {s}", .{ options.feature, @errorName(err) });
        std.process.exit(1);
    };
}

test "hypothesis and metrics" {
    const p = [_]LinearRegression.Point{ .{ .km = 0, .price = 1 }, .{ .km = 1, .price = 3 } };
    const m = LinearRegression.Model{ .theta0 = 1, .theta1 = 2 };
    try std.testing.expectEqual(@as(f64, 5), m.estimate(2));
    try std.testing.expectApproxEqAbs(@as(f64, 1), LinearRegression.metrics(&p, m).r2, 1e-12);
}
test "CSV and weights" {
    const p = try LinearRegression.parseCsv(std.testing.allocator, "km,price\n0,1\n1,3\n");
    defer std.testing.allocator.free(p);
    try std.testing.expectEqual(@as(usize, 2), p.len);
    const z = try LinearRegression.parseWeights("");
    try std.testing.expectEqual(@as(f64, 0), z.theta0);
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("bad"));
}

test "training exact line" {
    const p = [_]LinearRegression.Point{ .{ .km = 0, .price = 1 }, .{ .km = 1, .price = 3 }, .{ .km = 2, .price = 5 } };
    const m = try LinearRegression.train(&p, 1000, 0.1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1), m.theta0, 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 2), m.theta1, 1e-8);
}

test "plot clipping" {
    try std.testing.expectEqual(@as(usize, 0), LinearRegression.plotCoordinate(-1, 0, 10, 20));
    try std.testing.expectEqual(@as(usize, 19), LinearRegression.plotCoordinate(20, 0, 10, 20));
}

test "ordinary least squares reference recovers an exact line" {
    const points = [_]LinearRegression.Point{
        .{ .km = 1, .price = 3 },
        .{ .km = 2, .price = 5 },
        .{ .km = 3, .price = 7 },
    };

    const model = try LinearRegression.ordinaryLeastSquares(&points);

    try std.testing.expectApproxEqAbs(@as(f64, 1), model.theta0, 1e-12);

    try std.testing.expectApproxEqAbs(@as(f64, 2), model.theta1, 1e-12);
}

test "CSV rejects malformed and non-finite input" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.InvalidHeader, LinearRegression.parseCsv(a, "KM,price\n1,2\n"));
    try std.testing.expectError(error.InvalidRow, LinearRegression.parseCsv(a, "km,price\n1\n"));
    try std.testing.expectError(error.InvalidRow, LinearRegression.parseCsv(a, "km,price\n1,2,3\n"));
    try std.testing.expectError(error.InvalidRow, LinearRegression.parseCsv(a, "km,price\nnan,2\n"));
    try std.testing.expectError(error.EmptyDataset, LinearRegression.parseCsv(a, "km,price\n\n"));
}

test "CSV accepts CRLF blank lines and whitespace" {
    const p = try LinearRegression.parseCsv(std.testing.allocator, "  km,price  \r\n\r\n 1 , 2 \r\n");
    defer std.testing.allocator.free(p);
    try std.testing.expectEqual(@as(usize, 1), p.len);
    try std.testing.expectEqual(@as(f64, 1), p[0].km);
}

test "weight parser rejects every malformed shape" {
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("wrong\n1,2\n"));
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("theta0,theta1\n1\n"));
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("theta0,theta1\n1,2,3\n"));
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("theta0,theta1\nnan,2\n"));
    try std.testing.expectError(error.InvalidWeights, LinearRegression.parseWeights("theta0,theta1\n1,2\nextra\n"));
}

test "one epoch updates gradients simultaneously" {
    const p = [_]LinearRegression.Point{ .{ .km = 0, .price = 1 }, .{ .km = 1, .price = 3 }, .{ .km = 2, .price = 5 } };
    const m = try LinearRegression.train(&p, 1, 0.1, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 2.8), m.theta0, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.2), m.theta1, 1e-12);
}

test "constant price is valid and constant mileage is rejected" {
    const flat_y = [_]LinearRegression.Point{ .{ .km = 0, .price = 7 }, .{ .km = 1, .price = 7 } };
    const m = try LinearRegression.train(&flat_y, 100, 0.1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7), m.theta0, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0), m.theta1, 1e-9);
    const flat_x = [_]LinearRegression.Point{ .{ .km = 1, .price = 1 }, .{ .km = 1, .price = 2 } };
    try std.testing.expectError(error.ConstantMileage, LinearRegression.train(&flat_x, 10, 0.1, 0));
}

test "training rejects overflow and divergence" {
    const extreme = [_]LinearRegression.Point{ .{ .km = 1e308, .price = 1e308 }, .{ .km = -1e308, .price = -1e308 }, .{ .km = 0, .price = 0 } };
    try std.testing.expectError(error.UnusableScale, LinearRegression.train(&extreme, 10, 0.1, 0));
    const ordinary = [_]LinearRegression.Point{ .{ .km = 0, .price = 1 }, .{ .km = 1, .price = 3 }, .{ .km = 2, .price = 5 } };
    try std.testing.expectError(error.TrainingDiverged, LinearRegression.train(&ordinary, 10, 1e308, 0));
}

test "constant target R squared convention" {
    const p = [_]LinearRegression.Point{ .{ .km = 0, .price = 7 }, .{ .km = 1, .price = 7 } };
    try std.testing.expectEqual(@as(f64, 1), LinearRegression.metrics(&p, .{ .theta0 = 7 }).r2);
    try std.testing.expectEqual(@as(f64, 0), LinearRegression.metrics(&p, .{}).r2);
}

test "seeded shuffle is deterministic" {
    var a = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var b = a;
    var first = std.Random.DefaultPrng.init(42);
    var second = std.Random.DefaultPrng.init(42);
    first.random().shuffle(u8, &a);
    second.random().shuffle(u8, &b);
    try std.testing.expectEqualSlices(u8, &a, &b);
}
