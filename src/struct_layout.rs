use crate::ir::NameId;
use crate::program::Program;
use crate::type_inference::{
    ArrayType, BuiltinType, StructId, TypeId, TypeStore, TypeValue, UNKNOWN_FLOAT_SIZE,
    UNKNOWN_INT_SIZE, UNKNOWN_TYPE,
};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub size: usize,
    pub align: usize,
}

#[derive(Debug, Clone)]
pub struct FieldLayout {
    pub name: NameId,
    pub offset: usize,
    pub layout: Layout,
}

#[derive(Debug, Clone)]
pub struct StructLayout {
    pub size: usize,
    pub align: usize,
    pub fields: Vec<FieldLayout>,
}

#[derive(Debug, Clone, Copy)]
pub struct TargetLayout {
    pub pointer_size: usize,
    pub pointer_align: usize,
    pub int_size: usize,
    pub int_align: usize,
    pub uint_size: usize,
    pub uint_align: usize,
    pub str_size: usize,
    pub str_align: usize,
    pub fn_ptr_size: usize,
    pub fn_ptr_align: usize,
}

impl TargetLayout {
    pub fn native() -> Self {
        let pointer_size = std::mem::size_of::<usize>();
        let pointer_align = std::mem::align_of::<usize>();
        let int_size = pointer_size;
        let int_align = pointer_align;
        let uint_size = pointer_size;
        let uint_align = pointer_align;
        let str_size = pointer_size * 2;
        let str_align = pointer_align;
        let fn_ptr_size = pointer_size;
        let fn_ptr_align = pointer_align;

        Self {
            pointer_size,
            pointer_align,
            int_size,
            int_align,
            uint_size,
            uint_align,
            str_size,
            str_align,
            fn_ptr_size,
            fn_ptr_align,
        }
    }

    pub fn for_pointer_width(pointer_bits: usize) -> Option<Self> {
        let pointer_size = match pointer_bits {
            16 => 2,
            32 => 4,
            64 => 8,
            128 => 16,
            _ => return None,
        };
        let pointer_align = pointer_size;
        let int_size = pointer_size;
        let int_align = pointer_align;
        let uint_size = pointer_size;
        let uint_align = pointer_align;
        let str_size = pointer_size * 2;
        let str_align = pointer_align;
        let fn_ptr_size = pointer_size;
        let fn_ptr_align = pointer_align;

        Some(Self {
            pointer_size,
            pointer_align,
            int_size,
            int_align,
            uint_size,
            uint_align,
            str_size,
            str_align,
            fn_ptr_size,
            fn_ptr_align,
        })
    }
}

#[derive(Debug, Clone)]
pub enum LayoutError {
    RecursiveStruct {
        struct_id: StructId,
        field: Option<NameId>,
        cycle: Vec<StructId>,
    },
    UnsupportedType {
        type_id: TypeId,
    },
    UnknownType {
        type_id: TypeId,
    },
}

impl LayoutError {
    pub fn message(&self, program: &Program, store: &TypeStore) -> String {
        match self {
            LayoutError::RecursiveStruct {
                struct_id,
                field,
                cycle,
            } => {
                let struct_name = struct_label(program, store, *struct_id);
                let field_label = field
                    .map(|name| format!(" via field `{}`", program.name_string(name)))
                    .unwrap_or_default();
                let cycle_label = if cycle.is_empty() {
                    String::new()
                } else {
                    let path = cycle
                        .iter()
                        .map(|sid| struct_label(program, store, *sid))
                        .collect::<Vec<_>>()
                        .join(" -> ");
                    format!(" (cycle: {path})")
                };

                format!("recursive struct layout for `{struct_name}`{field_label}{cycle_label}")
            }
            LayoutError::UnsupportedType { type_id } => {
                let ty = store.get_type_string(program, *type_id);
                format!("cannot compute layout for unsupported type `{ty}`")
            }
            LayoutError::UnknownType { type_id } => {
                if *type_id == UNKNOWN_TYPE {
                    "cannot compute layout for unknown type".to_string()
                } else if *type_id == UNKNOWN_INT_SIZE {
                    "cannot compute layout for unknown int size".to_string()
                } else if *type_id == UNKNOWN_FLOAT_SIZE {
                    "cannot compute layout for unknown float size".to_string()
                } else {
                    "cannot compute layout for unknown type".to_string()
                }
            }
        }
    }
}

pub fn layout_type(
    store: &TypeStore,
    target: TargetLayout,
    type_id: TypeId,
) -> Result<Layout, LayoutError> {
    LayoutComputer::new(store, target).layout_type(type_id)
}

pub fn layout_struct(
    store: &TypeStore,
    target: TargetLayout,
    type_id: TypeId,
) -> Result<StructLayout, LayoutError> {
    LayoutComputer::new(store, target).layout_struct(type_id)
}

struct LayoutComputer<'a> {
    store: &'a TypeStore,
    target: TargetLayout,
    cache: HashMap<TypeId, StructLayout>,
    visiting: Vec<TypeId>,
}

impl<'a> LayoutComputer<'a> {
    fn new(store: &'a TypeStore, target: TargetLayout) -> Self {
        Self {
            store,
            target,
            cache: HashMap::new(),
            visiting: Vec::new(),
        }
    }

    fn layout_type(&mut self, type_id: TypeId) -> Result<Layout, LayoutError> {
        self.layout_type_inner(type_id, None, &[])
    }

    fn layout_type_inner(
        &mut self,
        type_id: TypeId,
        field: Option<NameId>,
        generics: &[TypeId],
    ) -> Result<Layout, LayoutError> {
        if type_id == UNKNOWN_TYPE || type_id == UNKNOWN_INT_SIZE || type_id == UNKNOWN_FLOAT_SIZE {
            return Err(LayoutError::UnknownType { type_id });
        }

        match self.store.type_value(type_id) {
            TypeValue::Builtin(builtin) => self.layout_builtin(*builtin),
            TypeValue::Tuple(items) => self.layout_tuple(items, generics),
            TypeValue::Func { generics, .. } => {
                if *generics != 0 {
                    return Err(LayoutError::UnsupportedType { type_id });
                }
                Ok(Layout {
                    size: self.target.fn_ptr_size,
                    align: self.target.fn_ptr_align,
                })
            }
            TypeValue::Ptr { tgt, .. } => {
                let is_unsized_array = matches!(
                    self.store.type_value(*tgt),
                    TypeValue::Array(_, ArrayType::Unsized)
                );
                let size = if is_unsized_array {
                    self.target.pointer_size.saturating_mul(2)
                } else {
                    self.target.pointer_size
                };
                Ok(Layout {
                    size,
                    align: self.target.pointer_align,
                })
            }
            TypeValue::Generic(gid) => {
                let Some(mapped) = generics.get(gid.0) else {
                    return Err(LayoutError::UnsupportedType { type_id });
                };
                if *mapped == type_id {
                    return Err(LayoutError::UnsupportedType { type_id });
                }
                self.layout_type_inner(*mapped, field, generics)
            }
            TypeValue::Struct { .. } => {
                let layout = self.layout_struct_with_field(type_id, field)?;
                Ok(Layout {
                    size: layout.size,
                    align: layout.align,
                })
            }
            TypeValue::Array(t, size) => match size {
                ArrayType::Sized(n) => {
                    let mut base = self.layout_type_inner(*t, field, generics)?;
                    base.size = base.size.saturating_mul(*n);
                    Ok(base)
                }
                ArrayType::Unsized => Err(LayoutError::UnsupportedType { type_id }),
            },
        }
    }

    fn layout_struct(&mut self, type_id: TypeId) -> Result<StructLayout, LayoutError> {
        self.layout_struct_with_field(type_id, None)
    }

    fn layout_struct_with_field(
        &mut self,
        type_id: TypeId,
        field: Option<NameId>,
    ) -> Result<StructLayout, LayoutError> {
        let (struct_id, generics) = match self.store.type_value(type_id) {
            TypeValue::Struct { id, generics, .. } => (*id, generics.as_slice()),
            _ => return Err(LayoutError::UnsupportedType { type_id }),
        };

        if let Some(existing) = self.cache.get(&type_id) {
            return Ok(existing.clone());
        }
        if self.visiting.contains(&type_id) {
            return Err(LayoutError::RecursiveStruct {
                struct_id,
                field,
                cycle: self.cycle_path(type_id),
            });
        }

        self.visiting.push(type_id);
        let rep = self.store.struct_value(struct_id);
        let mut offset = 0usize;
        let mut align = 1usize;
        let mut fields_layout = Vec::with_capacity(rep.fields.len());

        for (name, type_id) in rep.fields.iter() {
            let field_layout = self.layout_type_inner(*type_id, Some(*name), generics)?;
            offset = align_up(offset, field_layout.align);
            fields_layout.push(FieldLayout {
                name: *name,
                offset,
                layout: field_layout,
            });
            offset = offset.saturating_add(field_layout.size);
            align = align.max(field_layout.align);
        }

        let size = align_up(offset, align);
        let layout = StructLayout {
            size,
            align,
            fields: fields_layout,
        };
        self.visiting.pop();
        self.cache.insert(type_id, layout.clone());
        Ok(layout)
    }

    fn layout_tuple(
        &mut self,
        items: &[TypeId],
        generics: &[TypeId],
    ) -> Result<Layout, LayoutError> {
        let mut offset = 0usize;
        let mut align = 1usize;
        for item in items.iter().copied() {
            let item_layout = self.layout_type_inner(item, None, generics)?;
            offset = align_up(offset, item_layout.align);
            offset = offset.saturating_add(item_layout.size);
            align = align.max(item_layout.align);
        }
        Ok(Layout {
            size: align_up(offset, align),
            align,
        })
    }

    fn layout_builtin(&self, builtin: BuiltinType) -> Result<Layout, LayoutError> {
        let layout = match builtin {
            BuiltinType::Int => Layout {
                size: self.target.int_size,
                align: self.target.int_align,
            },
            BuiltinType::Uint => Layout {
                size: self.target.uint_size,
                align: self.target.uint_align,
            },
            BuiltinType::I8 | BuiltinType::U8 | BuiltinType::Bool => Layout { size: 1, align: 1 },
            BuiltinType::I16 | BuiltinType::U16 => Layout { size: 2, align: 2 },
            BuiltinType::I32 | BuiltinType::U32 | BuiltinType::F32 => Layout { size: 4, align: 4 },
            BuiltinType::I64 | BuiltinType::U64 | BuiltinType::F64 => Layout { size: 8, align: 8 },
            BuiltinType::I128 | BuiltinType::U128 => Layout {
                size: 16,
                align: 16,
            },
            BuiltinType::Isize | BuiltinType::Usize => Layout {
                size: self.target.pointer_size,
                align: self.target.pointer_align,
            },
            BuiltinType::Str => Layout {
                size: self.target.str_size,
                align: self.target.str_align,
            },
            BuiltinType::Void => Layout { size: 0, align: 1 },
            BuiltinType::Type => {
                return Err(LayoutError::UnsupportedType {
                    type_id: BuiltinType::Type.into(),
                });
            }
        };
        Ok(layout)
    }

    fn cycle_path(&self, type_id: TypeId) -> Vec<StructId> {
        let struct_id = match self.store.type_value(type_id) {
            TypeValue::Struct { id, .. } => *id,
            _ => return Vec::new(),
        };

        if let Some(pos) = self.visiting.iter().position(|tid| *tid == type_id) {
            let mut path = self.visiting[pos..]
                .iter()
                .filter_map(|tid| match self.store.type_value(*tid) {
                    TypeValue::Struct { id, .. } => Some(*id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            path.push(struct_id);
            path
        } else {
            vec![struct_id]
        }
    }
}

fn struct_label(program: &Program, store: &TypeStore, sid: StructId) -> String {
    match store.struct_value(sid).name {
        Some(name) => program.name_string(name).to_string(),
        None => format!("struct#{}", sid.0),
    }
}

fn align_up(value: usize, align: usize) -> usize {
    if align <= 1 {
        return value;
    }
    let mask = align - 1;
    (value + mask) & !mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::StructLayoutSpec;
    use crate::program::Program;
    use crate::type_inference::Nullable;
    use crate::type_inference::PointerStyle;
    use crate::type_inference::{ArrayType, GenId, StructRep, TypeStore, TypeValue};

    fn name(program: &mut Program, text: &str) -> NameId {
        let id = program.str_intern.intern(text);
        program.insert_value_in_global_scope(id)
    }

    #[test]
    fn layout_cycle_multi_element() {
        let mut program = Program::new();
        let mut store = TypeStore::new();

        let a_b = name(&mut program, "b");
        let a_x = name(&mut program, "x");
        let b_c = name(&mut program, "c");
        let c_a = name(&mut program, "a");

        let (a_sid, a_tid) =
            store.simple_struct(None, vec![(a_b, UNKNOWN_TYPE), (a_x, UNKNOWN_TYPE)]);
        let (b_sid, b_tid) = store.simple_struct(None, vec![(b_c, UNKNOWN_TYPE)]);
        let (c_sid, c_tid) = store.simple_struct(None, vec![(c_a, UNKNOWN_TYPE)]);

        store.set_struct_fields(a_sid, vec![(a_b, b_tid), (a_x, BuiltinType::Int.into())]);
        store.set_struct_fields(b_sid, vec![(b_c, c_tid)]);
        store.set_struct_fields(c_sid, vec![(c_a, a_tid)]);

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let err = layout_struct(&store, target, a_tid).unwrap_err();
        match err {
            LayoutError::RecursiveStruct {
                struct_id,
                field,
                cycle,
            } => {
                assert_eq!(struct_id, a_sid);
                assert_eq!(field, Some(c_a));
                assert!(cycle.iter().any(|sid| *sid == a_sid));
                assert!(cycle.iter().any(|sid| *sid == b_sid));
                assert!(cycle.iter().any(|sid| *sid == c_sid));
                assert!(cycle.len() >= 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn layout_pointer_cycle_allowed() {
        let mut program = Program::new();
        let mut store = TypeStore::new();

        let next = name(&mut program, "next");
        let (a_sid, a_tid) = store.simple_struct(None, vec![(next, UNKNOWN_TYPE)]);

        let ptr_a = store.intern(TypeValue::Ptr {
            tgt: a_tid,
            style: PointerStyle::Raw(Nullable::Yes),
            mutable: true,
        });
        store.set_struct_fields(a_sid, vec![(next, ptr_a)]);

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let layout = layout_struct(&store, target, a_tid).unwrap();
        assert_eq!(layout.size, 8);
        assert_eq!(layout.align, 8);
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].layout.size, 8);
    }

    #[test]
    fn layout_struct_field_raw_ptr_to_unsized_array_is_fat_pointer() {
        let mut program = Program::new();
        let mut store = TypeStore::new();

        let data = name(&mut program, "data");
        let (sid, tid) = store.simple_struct(None, vec![(data, UNKNOWN_TYPE)]);

        let unsized_array = store.intern(TypeValue::Array(
            BuiltinType::I32.into(),
            ArrayType::Unsized,
        ));
        let ptr_unsized = store.intern(TypeValue::Ptr {
            tgt: unsized_array,
            style: PointerStyle::Raw(Nullable::Yes),
            mutable: true,
        });
        store.set_struct_fields(sid, vec![(data, ptr_unsized)]);

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let layout = layout_struct(&store, target, tid).unwrap();

        assert_eq!(layout.size, 16);
        assert_eq!(layout.align, 8);
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].layout.size, 16);
        assert_eq!(layout.fields[0].layout.align, 8);
    }

    #[test]
    fn layout_zero_size_struct() {
        let mut store = TypeStore::new();
        let (_sid, tid) = store.simple_struct(None, Vec::new());

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let layout = layout_struct(&store, target, tid).unwrap();
        assert_eq!(layout.size, 0);
        assert_eq!(layout.align, 1);
        assert!(layout.fields.is_empty());
    }

    #[test]
    fn layout_generic_struct_specialized() {
        let mut program = Program::new();
        let mut store = TypeStore::new();

        let value = name(&mut program, "value");
        let generic = store.intern(TypeValue::Generic(GenId(0)));
        let rep = StructRep {
            name: None,
            fields: vec![(value, generic)],
            gen_count: 1,
            lifetime_params: Vec::new(),
            layout: StructLayoutSpec::Hot,
        };
        let sid = store.new_struct(rep);

        let specialized = store.intern(TypeValue::Struct {
            id: sid,
            generics: vec![BuiltinType::I32.into()],
            lifetimes: Vec::new(),
        });

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let layout = layout_struct(&store, target, specialized).unwrap();
        assert_eq!(layout.size, 4);
        assert_eq!(layout.align, 4);
        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.fields[0].layout.size, 4);
    }

    #[test]
    fn layout_generic_specialization_cycle() {
        let mut program = Program::new();
        let mut store = TypeStore::new();

        let value = name(&mut program, "value");
        let generic = store.intern(TypeValue::Generic(GenId(0)));
        let rep = StructRep {
            name: None,
            fields: vec![(value, generic)],
            gen_count: 1,
            lifetime_params: Vec::new(),
            layout: StructLayoutSpec::Hot,
        };
        let sid = store.new_struct(rep);

        let self_ty = store.intern(TypeValue::Struct {
            id: sid,
            generics: Vec::new(),
            lifetimes: Vec::new(),
        });
        store.values[self_ty.0] = TypeValue::Struct {
            id: sid,
            generics: vec![self_ty],
            lifetimes: Vec::new(),
        };

        let target = TargetLayout::for_pointer_width(64).unwrap();
        let err = layout_struct(&store, target, self_ty).unwrap_err();
        match err {
            LayoutError::RecursiveStruct {
                struct_id,
                field,
                cycle,
            } => {
                assert_eq!(struct_id, sid);
                assert_eq!(field, Some(value));
                assert!(cycle.iter().any(|cycle_id| *cycle_id == sid));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
