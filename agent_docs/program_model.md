# Program Model (`src/program.rs`)

`Program` is the owning context for lowered IR, definitions, scopes, and deferred name/label resolution.

Think of it as the compiler session state for one parsed input stream.

## Responsibilities

- Store all lowered arenas:
  - `values: Vec<Value>`
  - `patterns: Vec<Pattern>`
  - `type_exprs: Vec<TypeExpr>`
- Track locations for diagnostics (`*_locs`).
- Manage scoped names (`scopes`) and global definitions (`definitions`).
- Hold pending unresolved references (`pending_names`).
- Hold struct member methods (`member_methods`).
- Track function-local labels/gotos (`function_labels`, `label_names`).
- Own the global `StringInterner`.

## Definitions and Global Namespace

`Defined` variants:

- `Func(FunctionSet)`
  - `declarations: Vec<ValId>` for `fn/cfn` without a body
  - `implementations: Vec<ValId>` for `fn/cfn` with a body
- `Type(TExpId)`
- `BuildinType(TypeValue)`
- `BuildinInterface(StrId)`
- `Macro(Macro)`
- `ToBeDefined` placeholder for unresolved global references

Builtins are inserted during `Program::new()` via `insert_builtin_types()` (implemented in `type_inference.rs`).

## Scope Stack and Name Resolution

Scope layout:

- `scopes[0]`: global scope.
- `scopes[1..]`: nested local scopes.

Resolution order in `resolve_name`:

1. innermost local -> outer local
2. global
3. if missing globally, create fresh global placeholder:
   - insert into global scope
   - set `definitions[name] = ToBeDefined`
   - record use site in `pending_names`

This allows forward global references and mutually-recursive top-level forms.

`check_pending_names()` is the final gate; it reports unresolved placeholders that never received a real definition.

## Definition Gathering Pipeline

`lower_all(parser)` repeatedly does:

1. `parse_with_macros` (parse statement + recursive macro expansion)
2. `gather_definition` (classify and lower as global definition or expression)

`gather_definition` supports:

- semicolon wrappers (`Expr::Postfix(";", ...)`) by unwrapping and recursing
- block wrappers by iterating items
- assignment definitions (`lhs = rhs`) via `handle_assignment`
- `type` declarations
- fallback expression lowering

## Assignment Handling Rules

`handle_assignment` enforces global-definition restrictions.

- Valid RHS for global definition:
  - `macro(...) { ... }`
  - `fn` / `cfn`
  - `struct` / `cstruct` / `enum` / `union`
- Other RHS forms are rejected with `ERR_EXPECTED_DEFINITION_VALUE`.

Repeated global assignment to an already-defined name is rejected (`RepeatedGlobalAssignment`) for non-function definitions.

Global-name validation failures (non-identifier LHS, invalid reassignments) are emitted immediately during `get_ident_for_global` and are not silently dropped by callers.

Function definitions are append-only: repeated `name = fn...` / `name = cfn...` entries are merged into the same `FunctionSet` instead of overwriting.

## Member Method Registration

Special LHS form: `StructName.method = fn(...) { ... }`

`try_member_method_lhs` accepts only direct dotted identifiers and checks:

- base is a globally defined struct type name
- RHS is `fn` or `cfn`
- method name does not collide with struct field names
- repeated method definitions are merged into a `FunctionSet` (same split as globals: declarations vs implementations)

Stored shape: `member_methods[struct_name_id][method_name] = FunctionSet`.

## Label/Goto Patching Model

`Program` owns function-local label state (`function_labels` stack).

- Enter function body: `with_function_labels` pushes a fresh label map.
- `goto` use:
  - if label not defined yet, create or update `PendingLabel` and return `LabelId::PENDING`
- label definition:
  - marks `defined_loc`
  - rewrites all pending goto IR nodes to concrete `LabelId`
- function exit:
  - unresolved labels (used but never defined) produce `CompileError::UnresolvedLabel`

Outside a function label scope, both label definitions and gotos are rejected.

## Arena Convenience APIs

`id_*`, `set_*`, `reserve_*_span`, `*_loc` methods provide the sole path for mutation/readback of arena data and location metadata.

The reserve-then-overwrite pattern is relied on by lowering for contiguous sub-node storage.

## Practical Extension Notes

- If you add a new global-definition kind, update `handle_assignment` and likely `Defined`.
- If you add new unresolved flows, make sure they are checked in a finalization pass similar to `check_pending_names` or `with_function_labels`.
- Keep scope writes explicit (`insert_value_in_current_scope` vs `insert_value_in_global_scope`) to avoid accidental global leaks.
