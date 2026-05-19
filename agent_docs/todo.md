# Agent TODO Notes

## Type System Refactor

- Treat the existing type system as legacy reference material.
- Use the detached snapshot at
  `/home/user/Desktop/rust_stuff/expr_parser/expr_parse_pre` for old
  implementation details.
- Build the new system around ID-based shape solving, pending requirements, and
  late obligations.
- Recreate type dumps early because they are the most useful debugging tool.
- Keep old tests as behavioral references where the behavior is still intended.

## Refactor Milestones

1. Define the new ID-based type/shape/pointer-info data model.
2. Implement basic shape equality solving for literals, annotations, functions,
   structs, tuples, arrays, and pointers.
3. Add pending requirements for calls, operators, member access, indexing,
   deref, and specialization.
4. Add obligation recording for mutability, rawness, references, implicit
   derefs, reborrows, and lifetime relationships.
5. Add late pointer-style defaulting and obligation checks.
6. Rebuild lifetime graph/SCC solving on top of the new obligation/origin model.
7. Move borrow/storage checking toward a middle IR.
