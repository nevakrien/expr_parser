# expr_parser
this languge is kinda of ridiclous its essentially a C like grammar that treats everything as an expression.
so this
```
x = while t v; 
```

is a valid expression and so is
```
x = y = z = 2 = 3 = if x y else {z w}
```

";" "," and "(" are almost completly optional so the grammar is allowed to kinda go nuts.

the AST is fairly simplistic on purpose which should mean most functions on it are fairly small.
we are destinguishing between prefix postfix and infix operations to allow using just the operator string directly.

it should be fairly straight forward to add operators and behivior as the AST requires no design changes.

the main issue is that later you would still need to run a few checks on the outputs because some operators dont really make sense in some places, and there is no enum for them.


# syntax
for defining a varible we use a `let` expression, this semantically states that some address must be reserved even if the variable is constant.

```
let x = 5;
let y : int = 2
```

`let` expressions return the defined variable.
this can be used inside of loops like so
```
while(let t = next_token()){
	func(t)
}
```
note that that () and {} here are entirly optional.

similar to `let`, `while` and `if` are also expressions that return a value.
`while` returns the value of the last checked condition. this can be used to detect premature exit.
`if` returns the value of the chosen branch, if there is no `else` an Option is returned instead.
```
let x : Option[int] = if cond 4;
let x : int = if cond 4 else 5;
let x : bool = while x {
	x--
	if(x%2 == 0) break;
}
```

defining functions is fairly straight forward. they are a value like any other value.
they are also allowed to be declared globaly like so.

```
f = fn[T] (x:T)->T {
	return x
}
```

or predclared by ommiting the body. function types are the same as predclartions in terms of syntax.

similar to functions structs enums and unions are just type values to be assigned.
they all share the exact same syntax for construction

```
Point = struct[f] {x:f,b:f};
Point = struct[f] {x:f,b};
Point = struct[f] {x:f b};
```

construction of a struct/union can be done like so
```
Point{4,y=2}
Union{float=2.1}
```
note that when using a call/constructor name=x expressions are interpeted as passing arguments by name.

defining dot methods destructors and constructors are viewed as just operator overloading.
so constructing a class for point would look like
```
Point.dist = fn(self,other:Point)->float {
	let dx = self.x-other.x
	let dy = self.y-other.y
	sqrt(dx*dx+dy*dy)
}
drop(Point) = fn(self) {}
Point() = fn()->Point {Point{0,0}}
```

# Type System
currently we have basic generics and automatic infrence, and we would have some level of operator overloading.
with everything being required to be monomorphic in the end so we can get C++/Rust level inlining everywhere.

not all the features are implemented but we are slowly adding things to the languge.

we use hindly-miller and bi-directional typing where casts borrowing etc are checked after being infered.
this keeps the system simple enough with union-find rules. for example all casts are assumed to always be legal for unioning purposes, and they are later verified to be what we think it should be.

this already lets us have a smart pointer with something like this 
```
Box = struct[T]{ptr:*T};
Box.__free = fn[T](b:&mut Box[T]){free(b->ptr as *void)}
Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}
```
and we can do borrow checking on this and lower to something with destructive moves.

we can even add proper full overloading to this using the ideas from https://dl.acm.org/doi/10.1145/3763168 
which would boil down to making an empty type to throw at hindly miller. and then verifiying the overloads it could be later.

also note that traits like things can potentially be implemented like C++ templates so for example.
```
clone=fn[T:_](x:&T)->T;
clone=fn[int](x:&int)->int *x
Vec.clone = fn[T:clone,Vec[T]](v:&Vec[T])->Vec[T]{...}
```
which because they all share the same generic signature. we can solve clone as if it works for all T.
then later verify that clone indeed exists for the type we are cloning. and if it is not we emit a type error.

# Performance
this should be more than fast enough for any reasonbly size toy project. but it is still much slower than what is possible.

## benchmarks
### parser
- generate a large benchmark file:
  - `cargo run --release --example generate_benchmark_data -- benchmark_data.txt 1000000 20`
- run the file benchmark (stores expressions in a vector):
  - `cargo run --release --example file_parser_benchmark -- benchmark_data.txt`
- run the lexer peek benchmark:
  - `cargo run --release --example lexer_peek_benchmark`

recent profiling notes (file-based benchmark on a ~1M expression input):
- baseline throughput (dropping expressions immediately) ~0.75–0.78M expressions/sec on this machine.
- retaining expressions in a vector drops throughput to ~0.58M expressions/sec and raises LLC miss rate to ~71% (cpu_core).
- cache misses are dominated by `core::str::from_utf8` while validating the mmap'd input buffer.
- parser hot spots (`Parser::try_expr_bp`, `Lexer::lex_token`) show measurable misses but far below the utf-8 scan.
- branch miss rate remains low (<=~2% on cpu_core in both modes).

follow-up cache-miss breakdown using the `include_str!` variant (no runtime utf8 validation):
- cpu_core misses: 58.44% in libc free/allocator paths (drop teardown), 40.97% in binary drop paths for IR/Program data.
- cpu_atom misses: 100% in newline scanning (`lines()` -> `memchr` family).
- take-away: even after removing the dominant utf8 validation, almost none of the cache misses land in the main parse/lower logic; they are almost entirely in unavoidable cold input scanning and in drops/allocator teardown of the IR.

### macros
we started benchmarking macros before the rest of the compilation was made.
this lets us see how they compare to regular parsing and what cache behivior they show.

we see that macro expansion is still slower than the no-macro parser.
recent runs (macro_expansion_benchmark vs no_macros_benchmark) show ~1.23M expr/s vs ~3.93M expr/s and the data is very diffrent so this isnt necirally super meaningful.
mutating macro substitution in place did not hold up, so we kept the original apply/substitute flow.
regardless of if we have or dont have macros there is about 40% of setup/cleanup work.
the remaining 60% is split diffrently with macros sometimes dominating the work.

even in the most agrigous cases of macros dominating the work its never completly out takes parsing.
at most beating it to around 5x ish.

hard data:

we can find 6 chunks of work
1. parsing (`Parser::try_expr_bp` + `Parser::parse_token`)
2. macro core (`expand_macros_recursive` + `Macro::substitute_expr`)
3. allocator churn (malloc/free family)
4. control-flow glue (`ProgramParser::consume_expr` + `Parser::parse_stmt` + `Parser::parse_after_fn`)
5. clone/vec churn (`Vec::clone` + `Located::clone` + `Expr::clone`)
6. hash/lookup (`BuildHasher::hash_one` + `HashMap::insert`)

comparing macros to no macro cases we see this (on perf 6.8.12 on 13th Gen Intel(R) Core(TM) i9-13900K)

- cache miss rate from `perf stat`: no-macros 11.56% (cpu_core) / 52.61% (cpu_atom), macros 8.83% (cpu_core) / 37.77% (cpu_atom)

- cpu_core share from `perf record`:
  - parse:         no-macros 54.74%, macros 15.09% (`Parser::try_expr_bp` + `Parser::parse_token`)
  - macro core:    no-macros 4.50%,  macros 11.63% (`expand_macros_recursive` + `Macro::substitute_expr`)
  - alloc/free:    no-macros ~12.6%, macros ~23.9% (malloc/free family)
  - control-flow:  no-macros ~11.0%, macros ~3.7% (`ProgramParser::consume_expr` + `Parser::parse_stmt` + `Parser::parse_after_fn`)
  - clone/vec:     no-macros ~0%,    macros ~22.2% (`map::try_fold` + `Vec::from_iter` + `try_process`)
  - hash/lookup:   no-macros ~0%,    macros ~1.6% (`BuildHasher::hash_one` + `HashMap::insert`)

## adding preparsing of names
adding parsing of names before we try and lower, allowing compiling things like
```
f = fn(){
	g()
}

g = fn () {
	f()
}
```

compltly wrecked the preformance of our benchmarks, what used to basically take the same amount of time as just runing macros is now taking 2x or 5x longer.

### compilation algorithm improvement
after the name gathering feature was added, we made an algorithmic improvement to the compilation process that improved performance significantly.

benchmark results (10 iterations, ~100K statements):
- **old algorithm**: 1.555s ± 0.020s, 64,306 ± 811 stmts/sec
- **new algorithm**: 1.479s ± 0.007s, 67,605 ± 335 stmts/sec  
- **improvement**: 4.89% faster, 5.13% more throughput

key metrics:
- **5.09% fewer instructions** executed (17.60B → 16.71B)
- **8.5% fewer CPU cycles** (11.40B → 10.43B)
- **cache miss rate improved** from 74.26% to 72.88%
- **branch miss rate slightly worsened** from 1.47% to 1.55%

the improvements are statistically significant with much tighter variance on the new algorithm.

### flattening the ir tree
after flatterning the ir tree we cut from around 13-12 ms to 11-10 which is a semi signigicant imrpvoment. 
this was while till keeping some heap


### typechecker benchmarks
cargo run --release --example generate_typecheck_benchmark_data -- typecheck_benchmark_data.txt 100000
cargo run --release --example file_typecheck_benchmark -- typecheck_benchmark_data.txt

recent profiling notes (file_typecheck_benchmark, ~100k lines, perf record --call-graph dwarf):
- before reserving global HashMaps:
  - throughput: ~19.35s total, ~5.17k lines/sec, ~860 funcs/sec on this machine.
  - top hot spot is `hashbrown::HashMap::insert` at ~44% of samples, reached from `type_inference::finalize` when inserting into SolvedTypes.
  - `ariadne::Source` construction and line/utf8 scanning account for ~23% (string splitting + collection) plus ~3-4% in utf8 validation.
  - takeaway: typecheck time is dominated by HashMap insert paths and input/source prep rather than the inference rules themselves.

- after reserving global HashMaps:
  - throughput: 3.919s total, ~25.5k lines/sec, ~4.25k funcs/sec, ~39.19us/line on this machine.
  - speedup: ~4.94x faster overall, ~4.93x higher throughput vs the previous run.
  - cpu_core samples: `type_inference::main_solver` ~49.7%, `hashbrown::HashMap::insert` ~19.1%, `type_inference::find_root` ~18.8%.
  - cpu_atom samples: `core::slice::memchr::memchr_aligned` ~58.1%, `clear_page_erms` ~21.5%, `__memmove_avx_unaligned_erms` ~20.4%.

trying simdutf8 seems to not really help with performance meaningfully.
possible fixes for the utf8 problem is move validation into parsing so we dont do as much wrok outside of it.
importantly this is NOT a cache miss because we are doing a

### context & rough comparisons

On this machine, the typechecker processes ~100k lines in ~19.3s
(~5.2k LOC/s, ~193µs/line, single-threaded, no codegen, whole-program HM-style inference).

Very rough single-threaded comparisons on similarly sized codebases:

- **TypeScript (tsc)**: ~5k–20k LOC/s (50–200µs/line) on large projects with
  structural typing and heavy inference.
- **Rust (rustc, cold build)**: ~3k–10k LOC/s (100–300µs/line) per crate when
  including type checking, trait solving, and borrow checking.
- **Clang (C++)**: ~30k–100k LOC/s (10–30µs/line) on non-template-heavy code;
  substantially slower on template-intensive workloads.
- **Go**: ~50k–200k LOC/s (5–20µs/line), reflecting a deliberately simple and
  mostly local type system.
- **TCC (Tiny C Compiler)**: historically reported at ~800k+ LOC/s on older
  hardware (~2.4 GHz Pentium 4), emphasizing minimal analysis and extremely
  fast code generation.

so not the best but not the worse either. we can see that languges with hindly miller style infrence are just slower.
this code is purposfully single threaded so not the most fair comperison to rustc that has multithreading.
