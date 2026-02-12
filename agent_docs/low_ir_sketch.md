# Low IR Sketch Status (`src/low_ir.rs`)

`src/low_ir.rs` is currently a design sketch, not an active compiler stage.

## Current State

- The module is small and marked `#![allow(dead_code)]`.
- It is not wired into parsing/lowering/type inference pipelines.
- No pass currently produces or consumes this IR.

## Intent

The comments describe a lower-level IR that is closer to direct backend lowering (LLVM/GIMPLE-like), with ownership/lifetime side effects made explicit.

Planned semantics called out in comments:

- explicit memory reads/writes
- explicit control flow (`Jump`, `Branch`)
- explicit ownership operations (`Drop`)
- borrow flavors (`Borrow`, `BorrowMut`, raw borrow variants)

## Draft Data Model

Identifiers:

- `Label(usize)`
- `IRV(usize)` for low-level values

`Operation` enum includes:

- `Write`, `Read`
- `Jump`, `Branch`
- `BitFidle(ValId)` placeholder for primitive operations
- `Call(ValId, Vec<IRV>)`
- `Drop(IRV)`
- borrow operations

The use of `ValId` references ties the sketch to high-level IR IDs for now.

## Gaps Before Integration

- no CFG/block container structure
- no pass from `Value` IR into `Operation`
- no representation for temporaries/liveness regions beyond raw IDs
- no error model or diagnostics mapping
- no tests

## Practical Recommendation

Treat this file as roadmap notes. If implementing ownership/destructor insertion or backend lowering, design should likely move from this sketch toward:

1. explicit basic blocks
2. stable low-level value IDs independent from high-level `ValId`
3. a dedicated lowering pass after type inference finalization
