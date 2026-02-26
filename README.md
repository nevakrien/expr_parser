# expr_parser
this languge is kinda of ridiclous its essentially a mix of Rust and C++ with a C like grammar that treats everything as an expression. the main goal of this languge is to be non restrictive to its users and support multiple styles.
its aimed at intermidate programers who want to learn system level stuff but not sure what style exactly they prefer.

we have optional borrow checking for programs using safe refrences, but using regular pointers or unsafe refrences is also supported just as much. its generally expected that both are used for most programs so people getting into systems programing can decide which style they like best.


# syntax

syntax errors on missing semi colons and simlar things are deliberatly ignored.

so this
```
x = while t v; 
```

is a valid expression and so is
```
x = y = z = 2 = 3 = if x y else {z w}
```

";" "," and "(" are almost completly optional so the grammar is allowed to kinda go nuts.

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

calling functions is allowed like so 
```
f(1,2)
1 |> f(2)
```
code generally doesnt mind too much which way ur going with (currently named args arent supported on functions but they will be).

similar to functions structs enums and unions are just type values to be assigned.
they all share the exact same syntax for construction

```
Point = struct[f] {x:f,b:f};
```

construction of a struct/union can be done like so
```
Point{4,y=2}
Union{float=2.1}
```
note that when using a call/constructor name=x expressions are interpeted as passing arguments by name.
this includes when they are piped as pipe expressions are essentially just reorgenizing.

defining dot methods destructors and constructors are viewed as just operator overloading.
so constructing a class for point would look like
```
sqrt = cfn(f:float)->float;
Point = struct{x:float,y:float}
Point.dist = fn(self:&Point,other:Point)->float {
    let dx = self.x-other.x
    let dy = self.y-other.y
    sqrt(dx*dx+dy*dy)
}
__user_free = fn(self:&mut Point) {}
Point.new = fn()->Point {Point{0.0,0.0}}
Point.__add = fn(p1:&Point,p2:Point)->Point {Point{p1.x+p2.x,p1.y+p2.y}}
f=fn(p1:Point,p2:Point)->Point {p1+p2}
```
notice we can overload the + method for Point in order to make nicer syntax.
type infrence very delibrately wont break when adding new operator overloads.
we basically assume all overloads may exist for all types untill the very last moment.


we also support generics so nad destructors to allow for this
```
Box = struct[T]{ptr:&'raw T};

free = cfn(p:*void);
no_fail_alloc = cfn(s:usize)->*void;
Box.new = fn[T](x:T)->Box[T] {
  let p=no_fail_alloc(x.__size_of());
  Box{p as &'raw _}
}
Box.__free = fn[T](b:&mut Box[T]){
(*b.ptr).__free()
free(b->ptr as *void)
}


Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}

f=fn(b:Box[[int]])->int { let y:int = b[0]; y };

```

is completly valid code. if we really wanted we could mark ptr as being &'static mut so that its passed as non aliasing.
but thats a bit more confusing and can cause some confusion when acessing ptr directly.

## Types
types are automatically infered with casts being as _ and :  meaning "this value is of type ..."
so for example
```
f=fn(x:int)->float {
  let y = 1.0+0.2+x as _
  sqrt(y:float)
}
```
this gurntees y is a float when calling sqrt, and we cast x from an int to a float.
lifetime

in type expressions pointers are assumed to be mutable while safe refrences are assumed constants so
```
*int //mut
*const int//const
*mut int//mut

&'raw int //mut
&'raw const int//const
&'raw mut int//mut

&int //const
&const int //const
&mut int //mut
```


## Lifetimes (planned semantics)
lifetime rules of safe refrences (& and &mut) follow the same underlying semantics of Rust.
ie 1 and only 1 holder of &mut. this is because operations like vector resize or free take in a &mut.
so for them to be safe it has to be unique. 
this also lets us put noalias/restrict on every safe refrences which is great for performance.

but we still want to support C++ like code that doesnt fit neatly into the lifetime model.
which is why we take kind of a middele road aproch.

we also do less implicit lifetime casts to what Rust does. partly as a limitation and partly as a design choice.
Rust would allow passing `&'a &'b` into a `&'a &'a` completly implictly. a similar sort of cast is actually required to exploit 1 of their worse [compiler bugs](https://github.com/Speykious/cve-rs/blob/main/src/lifetime_expansion.rs).

member methods can either use lifetimes or use raw refrences.
its on users to mark unsafe methods with unsafe_... or not we dont mind.
if code does not use any raw pointers it is gurnteed to safe (and if there is UB thats a compiler bug).
the main way this works is we have  &'raw which is essentially a non null pointer.
raw can be used in .methods to make raw pointer like behivior
```
Pointer.__deref_mut = fn['a](self:&'raw self)->&'a mut int {...}
```

or if we dont want to mess around the aliasing rules its possible to stay entirly within &'raw.
raw is never going to be infered by the compiler unless explictly stated in a method signature.
so it is not going to creep up on safe code.

if u do want no-alias then u have to take in a &mut and this is because no alias is just such a trap inherently.
u can get around this by immidiatly casting the &mut into a &raw in the functions body.
and by doing the explicit cast &* on the call site.
but chances are this is just a footgun around UB. no-alias is incredibly strict,

lifetimes are also more explicit from what they are in Rust and the system is just less powerfull 
rust has closures that are generic over lifetime and this actually allows a lot of powerful patterns like taging.

we also dont automatically coehrce lifetimes, instead an explicit reborrow is needed.
this is partly a limitation because coersion is hard but its also a philosphy of less implicit behivior.
we very delibratly keep . to a single implicit deref. for the rust style "deref until u find something"
users are expcted to use -> which is an explicit way to show more than 1 derfrence

## Safety and Panics
we dont have exceptions at all... this is a delibrate decision as they cause a lot of weird edge cases for usnafe code.
panic would simply be a trap instruction. and some code would also print the reason for panics in debug mode.

array indexing for example is bounds check with a trap and also automatically derfrences.
if users want a less implicit operation calling get() or direct pointer arithmetic is the way to go.


we will hopefully ship a sanitizer runtime that comes with the languge and checks that user code does actually fufil the requirments.


# Design 
the AST is fairly simplistic on purpose which should mean most functions on it are fairly small.
we are destinguishing between prefix postfix and infix operations to allow using just the operator string directly.

it should be fairly straight forward to add operators and behivior as the AST requires no design changes.

the main issue is that later you would still need to run a few checks on the outputs because some operators dont really make sense in some places, and there is no enum for them.

## Type Checking
currently we have basic generics and automatic inference,
with everything being required to be monomorphic in the end so we can get C++/Rust level inlining everywhere.

we use hindly-miller and bi-directional typing where casts borrowing etc are checked after being inferred.
this keeps the system simple enough with union-find rules. for example all casts are assumed to always be legal for unioning purposes, and they are later verified to be what we think it should be.

this already lets us have a smart pointer with something like this 
```


//user code
free = cfn(p:*void);
Box.__free = fn[T](b:&mut Box[T]){
(&*b.ptr).__free()
free(b->ptr as *void)
}

Box = struct[T]{ptr:*T};
Box.__deref = fn[T](b:&const Box[T])->&T{&*b.ptr}
Box.__deref_mut = fn[T](b:&mut Box[T])->&mut T{&*b.ptr}
```
and we can do borrow checking on this and lower to something with destructive moves.
notice that the example overrides `__free` as a member method on `Box[T]`.


for traits/templates we can have something like
```
clone=fn[T:_](x:&T)->T;
clone=fn(x:&int)->int *x
clone = fn[T:clone,Vec[T]](v:&Vec[T])->Vec[T]{...}
```
T:\_ means that clone is allowed to not be implemented for all T.
this naturally allows for faily powerful overloading which gets type checked like regular generics.

we can even add proper full overloading to this using the ideas from https://dl.acm.org/doi/10.1145/3763168 
but this has been avoided as it leads to confusing type checks we might allow overloading on arity alone.
this is because such overloads are trivial to resolve and when combined with generic methods they give the full range of functions.
the only real problem is that the type checker would be unable to infer a few things if everything is generic.

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

legacy generator (pre smart-deref/method benchmark mix):
cargo run --release --example generate_typecheck_benchmark_data_old -- typecheck_benchmark_data.txt 100000

originally wrote a pretty bad algorithmic bug of runing O(n) in a loop causing O(n^2).
after doing this we are up to 900k lines checked per second. at the start we were at 5k.
reserving the hashmap ahead of time got us to around 25k and moving to indecies moved us to 32k.
