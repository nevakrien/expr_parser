# Struct Layout Notes (`src/struct_layout.rs`)

This module computes size/alignment/field-offset layout from solved `TypeStore` types.

It is target-parameterized and intentionally independent from backend codegen.

## Public API

- `layout_type(store, target, type_id) -> Result<Layout, LayoutError>`
- `layout_struct(store, target, type_id) -> Result<StructLayout, LayoutError>`

`layout_type` works for any type that has a concrete runtime layout.
`layout_struct` requires `TypeValue::Struct { ... }` input.

## Target Model

`TargetLayout` carries primitive ABI numbers:

- pointer size/align
- default int/uint size/align
- `str` size/align (fat pointer style in this model)
- function pointer size/align

Helpers:

- `TargetLayout::native()` from host `usize`
- `TargetLayout::for_pointer_width(bits)` for fixed 16/32/64/128-bit pointer targets

## Core Algorithm

`LayoutComputer` does recursive traversal with:

- `cache: HashMap<TypeId, StructLayout>` for already-computed struct instances
- `visiting: Vec<TypeId>` for recursion detection/cycle reporting

Struct layout flow:

1. Validate input is a struct instance (`TypeValue::Struct`).
2. Detect recursion via `visiting.contains(type_id)`.
3. Iterate fields in declaration order.
4. Align current offset with `align_up(offset, field_align)`.
5. Record field offset and field layout.
6. Advance offset by field size.
7. Final struct size is `align_up(offset, max_align)`.

Tuple layout follows the same packing rules, but without field names.

## Generic Specialization Behavior

For `TypeValue::Generic(gid)` during layout:

- map through the enclosing struct instance generic argument slice
- recurse on mapped concrete type

If mapping is missing or invalid, layout returns `UnsupportedType`.

This allows layout of specialized struct instances (for example `Box[i32]`) without mutating the global type store.

## Recursion Rules

Direct by-value recursive structs are rejected:

- produces `LayoutError::RecursiveStruct { struct_id, field, cycle }`

Pointer indirection naturally breaks recursion because pointer layout is fixed-size; pointer-linked self types are accepted.

Cycle details are produced by `cycle_path` using `visiting` snapshot.

## Unsupported/Unknown Cases

`LayoutError::UnsupportedType` includes cases like:

- `TypeValue::WithGenerics { ... }`
- `BuiltinType::Type`
- malformed generic mapping

`LayoutError::UnknownType` is used for inference sentinels:

- `UNKNOWN_TYPE`
- `UNKNOWN_INT_SIZE`
- `UNKNOWN_FLOAT_SIZE`

`LayoutError::message(program, store)` converts these into user-facing text.

## Builtin Layout Policy

- Integer/float primitives use fixed widths.
- `int`/`uint` follow target defaults.
- `isize`/`usize` follow pointer width.
- `str` uses target `str_size`/`str_align`.
- function types layout as function pointers.
- pointers layout as pointer size/alignment, except pointers to unsized arrays (`*[T]`, `&[T]`, `&mut [T]`) which are modeled as fat pointers (`2 * pointer_size`, pointer alignment).
- `void` is zero-size align-1.

## Practical Extension Notes

- If type inference adds new `TypeValue` variants, `layout_type_inner` must be updated.
- If ABI policy changes (for example distinct extern fn-ptr ABI), update `TargetLayout` and builtin mapping.
- Keep recursion detection tied to concrete `TypeId` instances; this correctly handles generic specializations that may or may not recurse.
