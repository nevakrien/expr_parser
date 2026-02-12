//this module is IR that should be obvious to lower directly to LLVM or GIMPLE
//its also inserting destructors and borrow checks when lowering
#![allow(dead_code)]

use crate::ir::ValId;

struct Label(usize);
struct IRV(usize);

///currently sketched as tree form
///would be moved to id form when we got it
enum Operation {
    Write(IRV, IRV),
    Read(IRV),
    Jump(Label),
    Branch(IRV, Label),

    ///these are operation on basic copy types
    ///we truly dont need to add anything to this
    ///might be nice to do anyaway but for borrow check the answer is YES thats it
    BitFidle(ValId),

    ///calls take ownership of the values
    ///closure take full ownership of anything they capture (if you want it by ref do let x=&x; before closure)
    Call(ValId, Vec<IRV>),

    ///these are auto generated at end of scope for values that have __free
    ///expressions like x; are statments to drop a value now
    ///they are diffrent to calling an empty function
    ///because x; also explictly drops refrences asserting they arent used anymore
    Drop(IRV),

    Borrow(IRV),
    BorrowMut(IRV),
    BorrowRaw(IRV),
    BorrowRawConst(IRV),
}
