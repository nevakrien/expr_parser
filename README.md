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

";" and "(" are completly optional so the grammar is allowed to kinda go nuts.


the AST is fairly simplistic on purpose which should mean most functions on it are fairly small.
it should be fairly straight forward to add operators and behivior as the AST requires no design changes.

the main issue is that later you would still need to run a few checks on the outputs because some operators dont really make sense in some places.


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

