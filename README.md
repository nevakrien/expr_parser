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
