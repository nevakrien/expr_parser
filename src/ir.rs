// pub enum BasicOp{
// 	Assign,
// 	Deref,Adder,

// 	Call,Index,

// 	Add,Sub,Mul,Div,Mod,
// 	ShiftLeft,ShiftRight,

// 	//...
// }

// pub struct Value{
// 	id:ValueId,
// 	op:BasicOp,
// 	parts:Vec<ValueId>,
// }

// #[derive(Debug,Copy,Clone,PartialEq,Eq,Hash)]
// pub struct ValueId(u32);

// pub struct Block {
// 	id:ValueId,
// 	parts:Vec<Statment>,
// 	returns:Option<ValueId>,
// }

// pub enum Statment {
// 	VarDef{
// 		id:ValueId,
// 		//some info can go here
// 	},
// 	ValueOp{
// 		val:Value,
// 	},
// 	Block{
// 		inner:Block,
// 		tgt:ValueId,
// 	},
// 	If{
// 		tgt:ValueId,
// 		cond:ValueId,
// 		yes:Block,
// 		no:Block,
// 	},
// 	While{
// 		tgt:ValueId,

// 		cond:Block,
// 		body:Block,
// 	},
// 	Ret(Value),
// 	Break,
// 	Continue,
// }
