const assert = std.debug.assert;
const std = @import("std");
const InternPool = @import("../../InternPool.zig");
const Type = @import("../../Type.zig");
const Zcu = @import("../../Zcu.zig");

pub const Class = union(enum) {
    memory,
    byval,
    integer,
    double_integer,
    float_array: u8,
};

/// For `float_array` the second element will be the amount of floats.
pub fn classifyType(ty: Type, zcu: *Zcu) Class {
    assert(ty.hasRuntimeBits(zcu));

    switch (ty.zigTypeTag(zcu)) {
        .@"struct" => {
            if (ty.containerLayout(zcu) == .@"bitpack") return .byval;
            if (countFloats(ty, zcu)) |float| return .{ .float_array = float.count };

            const bit_size = ty.abiSize(zcu) * 8;
            if (bit_size > 128) return .memory;
            if (bit_size > 64) return .double_integer;
            return .integer;
        },
        .@"union" => {
            if (ty.containerLayout(zcu) == .@"bitpack") return .byval;
            if (countFloats(ty, zcu)) |float| return .{ .float_array = float.count };

            const bit_size = ty.abiSize(zcu) * 8;
            if (bit_size > 128) return .memory;
            if (bit_size > 64) return .double_integer;
            return .integer;
        },
        .int, .@"enum", .error_set, .bool => return .byval,
        .float => return switch (ty.floatBits(zcu.getTarget())) {
            else => unreachable,
            16, 32, 64, 128 => .byval,
            80 => .double_integer,
        },
        .vector => {
            const bit_size = ty.bitSize(zcu);
            // TODO is this controlled by a cpu feature?
            if (bit_size > 128) return .memory;
            return .byval;
        },
        .optional => {
            assert(ty.isPtrLikeOptional(zcu));
            return .byval;
        },
        .pointer => {
            assert(!ty.isSlice(zcu));
            return .byval;
        },
        .error_union,
        .frame,
        .@"anyframe",
        .noreturn,
        .void,
        .type,
        .comptime_float,
        .comptime_int,
        .undefined,
        .null,
        .@"fn",
        .@"opaque",
        .spirv,
        .enum_literal,
        .array,
        => unreachable,
    }
}

const CountFloatsResult = struct {
    ty: Type,
    count: std.math.IntFittingRange(0, max_count),

    const none: CountFloatsResult = .{ .ty = .void, .count = 0 };

    const max_count = 4;
};
/// Counts the members of a Homogeneous Floating-point Aggregate (HFA) or a
/// Homogeneous Short-Vector Aggregate (HVA), as defined by AAPCS64 5.9.5.
/// The base type is either a float or a 64/128-bit short vector, and arrays
/// count as their elements, so `struct { float4 columns[4]; }` (the layout of
/// `simd_float4x4`) is an HVA of four 128-bit vectors.
fn countFloats(ty: Type, zcu: *Zcu) ?CountFloatsResult {
    const ip = &zcu.intern_pool;
    if (!ty.hasRuntimeBits(zcu)) return .none;
    switch (ty.zigTypeTag(zcu)) {
        .@"union" => {
            const loaded_union = zcu.typeToUnion(ty).?;
            var result: CountFloatsResult = .none;
            for (loaded_union.field_types.get(ip)) |field_ty| {
                const float = countFloats(Type.fromInterned(field_ty), zcu) orelse return null;
                if (float.count == 0) continue;
                if (result.ty.toIntern() == .void_type) {
                    result.ty = float.ty;
                } else if (!sameBaseType(result.ty, float.ty, zcu)) return null;
                result.count = @max(result.count, float.count);
            }
            if (ty.abiSize(zcu) != result.ty.abiSize(zcu) * result.count) return null;
            return result;
        },
        .@"struct" => {
            var result: CountFloatsResult = .none;
            var field_it: InternPool.LoadedStructType.RuntimeOrderIterator = if (zcu.typeToStruct(ty)) |loaded_struct|
                loaded_struct.iterateRuntimeOrder(ip)
            else
                .{ .runtime_order = null, .fields_len = ty.structFieldCount(zcu), .next_index = 0 };
            while (field_it.next()) |field_index| {
                const field_ty = ty.fieldType(field_index, zcu);
                const float = countFloats(field_ty, zcu) orelse return null;
                if (float.count == 0) continue;
                if (ty.structFieldOffset(field_index, zcu) != result.ty.abiSize(zcu) * result.count) return null;
                if (result.ty.toIntern() == .void_type) {
                    result.ty = float.ty;
                } else if (!sameBaseType(result.ty, float.ty, zcu)) return null;
                if (float.count > CountFloatsResult.max_count - result.count) return null;
                result.count += float.count;
            }
            if (ty.abiSize(zcu) != result.ty.abiSize(zcu) * result.count) return null;
            return result;
        },
        .array => {
            const elem = countFloats(ty.childType(zcu), zcu) orelse return null;
            const len = ty.arrayLen(zcu);
            if (elem.count == 0 or len == 0) return .none;
            if (len > CountFloatsResult.max_count / elem.count) return null;
            return .{ .ty = elem.ty, .count = @intCast(elem.count * len) };
        },
        .vector => return switch (ty.abiSize(zcu)) {
            // Short vectors.
            8, 16 => .{ .ty = ty, .count = 1 },
            else => null,
        },
        .float => return .{ .ty = ty, .count = 1 },
        else => return null,
    }
}

/// Whether two HFA/HVA members have the same fundamental data type: floats of
/// the same size, or short vectors of the same size.
fn sameBaseType(a: Type, b: Type, zcu: *Zcu) bool {
    const a_tag = a.zigTypeTag(zcu);
    if (a_tag != b.zigTypeTag(zcu)) return false;
    return switch (a_tag) {
        .float => a.floatBits(zcu.getTarget()) == b.floatBits(zcu.getTarget()),
        .vector => a.abiSize(zcu) == b.abiSize(zcu),
        else => unreachable,
    };
}

pub fn getFloatArrayType(ty: Type, zcu: *Zcu) ?Type {
    const ip = &zcu.intern_pool;
    switch (ty.zigTypeTag(zcu)) {
        .@"union" => {
            const loaded_union = zcu.typeToUnion(ty).?;
            for (loaded_union.field_types.get(ip)) |field_ty| {
                if (getFloatArrayType(Type.fromInterned(field_ty), zcu)) |some| return some;
            }
            return null;
        },
        .@"struct" => {
            var field_it: InternPool.LoadedStructType.RuntimeOrderIterator = if (zcu.typeToStruct(ty)) |loaded_struct|
                loaded_struct.iterateRuntimeOrder(ip)
            else
                .{ .runtime_order = null, .fields_len = ty.structFieldCount(zcu), .next_index = 0 };
            while (field_it.next()) |field_index| {
                const field_ty = ty.fieldType(field_index, zcu);
                if (getFloatArrayType(field_ty, zcu)) |some| return some;
            }
            return null;
        },
        .array => return getFloatArrayType(ty.childType(zcu), zcu),
        .float, .vector => return ty,
        else => return null,
    }
}
