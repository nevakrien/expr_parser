use crate::type_system::TypeUniverse;

pub struct ReqResult {
    pub done: bool,
    pub progress: bool,
}

pub trait Requirment {
    fn try_resolve(&mut self, info: &mut TypeUniverse) -> ReqResult;
}
