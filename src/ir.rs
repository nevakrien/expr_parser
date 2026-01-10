// pub struct SimpleOp;
// pub struct Value;

// #[derive(Debug,Copy,Clone,PartialEq,Eq,Hash)]
// pub struct BlockId(u32);

// pub enum Block {
// 	Many{
// 		parts:Vec<BlockId>,
// 	},
// 	Basic{
// 		parts:Vec<SimpleOp>,
// 	},
// 	If{
// 		cond:Value,
// 		yes:BlockId,
// 		no:BlockId,
// 	},
// 	While{
// 		cond:Value,
// 		body:BlockId,
// 	},
// 	Ret(Value),
// }
