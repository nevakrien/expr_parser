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
f[T] = fn (x:T)->T {
	return x
}
```

or predclared by ommiting the body. function types are the same as predclartions in terms of syntax.

similar to functions structs enums and unions are just type values to be assigned.
they all share the exact same syntax for construction

```
Point[f] = struct {x:f,b:f};
Point[f] = struct {x:f,b};
Point[f] = struct {x:f b};
```

construction of a struct/union can be done like so
```
Point(4,y=2)
Union(float=2.1)
```

defining dot methods destructors and constructors are viewed as just operator overloading.
so constructing a class for point would look like
```
Point.dist = fn(self,other:Point)->float {
	let dx = self.x-other.x
	let dy = self.y-other.y
	sqrt(dx*dx+dy*dy)
}
drop(Point) = fn(self) {}
Point() = fn()->Point {Point(0,0)}
```

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

### macros
we started benchmarking macros before the rest of the compilation was made.
this lets us see how they compare to regular parsing and what cache behivior they show.

we see that macro expansion is still slower than the no-macro parser.
recent runs (macro_expansion_benchmark vs no_macros_benchmark) show ~0.99M expr/s vs ~3.68M expr/s note the data is very diffrent so this isnt necirally super meaningful.
regardless of if we have or dont have macros there is about 40% of setup/cleanup work.
the remaining 60% is split diffrently with macros sometimes dominating the work.

even in the most agrigous cases of macros dominating the work its never completly out takes parsing.
at most beating it to around 5x ish.

hard data:

we can find 5 chunks of work
1. parsing (`Parser::try_expr_bp` + `Parser::parse_token`)
2. macro core (`expand_macros_recursive` + `Macro::substitute_expr`)
3. allocator churn (malloc/free family)
4. control-flow glue (`ProgramParser::consume_expr` + `Parser::parse_stmt` + `Parser::parse_after_fn`)
5. iterator/vec plumbing (`map::try_fold` + `Vec::from_iter` + `try_process`)

comparing macros to no macro cases we see this (on perf 6.8.12 on 13th Gen Intel(R) Core(TM) i9-13900K)

- cache miss rate from `perf stat`: no-macros 14.56% (cpu_core) / 44.73% (cpu_atom), macros 7.54% (cpu_core) / 28.76% (cpu_atom)

- cpu_core share from `perf record`:
  - parse:         no-macros 44.49%, macros 11.81% (`Parser::try_expr_bp` + `Parser::parse_token`)
  - macro core:    no-macros 13.49%, macros 17.77% (`expand_macros_recursive` + `Macro::substitute_expr`)
  - alloc/free:    no-macros ~19.4%, macros ~31.9% (malloc/free family)
  - control-flow:  no-macros ~12.5%, macros ~3.2% (`ProgramParser::consume_expr` + `Parser::parse_stmt` + `Parser::parse_after_fn`)
  - iter/vec work: no-macros ~0%,    macros ~18.7% (`map::try_fold` + `Vec::from_iter` + `try_process`)

