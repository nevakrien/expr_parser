# Agent TODO Notes

## Nice To Have
currently TypeValue is expensive to clone making it not usable with .iter().map

causing a lot of explicit vec construction and pushing with
```
if let SomePattern{vars,..} = ex.store.type_value(ty) else {
	unreachble!()
}
```
in id based loops to avoid the issue.
if we moved all the expensive fields to be Rc<[]> that would allow us to fixup this code to be more readble
