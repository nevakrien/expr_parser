
use crate::type_system::LifeId;
use std::marker::PhantomData;
use crate::data_structures::graph::*;
use crate::data_structures::index::*;

pub struct LifeTimeInfo {
	///static is 0 and then all lifetimes are 1 after another
	pub contain_sets:SparseBits<CId<LifeId>,u32>,

	///points represent specific part of MIR this is not a finished API
	pub point_sets:SparseBits<CId<LifeId>,u32>,

	
	pub scc: SCCS<LifeId>,
}

// pub enum LifeTimeRep {
// 	Universal(u32),
// 	Local(CId<LifeId>),
// }

// impl LifeTimeInfo {
// 	pub fn get_life_of(&self,id:LifeId)->LifeTimeRep{

// 	}
// }

#[derive(Debug,Clone)]
pub struct BS<Node:Idx>{
	storage:Box<[u64]>,
	_mark:PhantomData<fn(Node)->bool>
}

impl<Node:Idx> BS<Node> {
	pub fn new(size:usize)->Self{
		let s = (size+63)/64;
		let storage = (0..s).map(|_| 0).collect();
		Self{
			storage,
			_mark:PhantomData,
		}
	}

	pub fn new_ones(size:usize)->Self{
		let mut ans = Self::new(size);
		for i in 0..size {
			ans.insert(Node::new(i));
		}
		ans
	}

	pub fn clear(&mut self){
		for r in self.storage.iter_mut(){
			*r=0;
		}
	}

	pub fn insert(&mut self,node:Node){
		let idx = node.index();
		let base = idx/64;
		let offset = idx%64;
		let v = &mut self.storage[base];
		*v= *v | 1<<offset;
	}

	pub fn contains(&self,node:Node)->bool{
		let idx = node.index();
		let base = idx/64;
		let offset = idx%64;
		(self.storage[base] & 1<<offset)!=0
	}

	pub  fn merge_with(&mut self,other:&Self){
		for (u1,u2) in self.storage.iter_mut().zip(other.storage.iter()){
			*u1=*u1 | * u2;
		}
	}
}

pub struct SparseBits<Row:Idx,Col:Idx>{
	stroage: IndexVec<Row,Option<BS<Col>>>,
	size:usize
}

impl<Row:Idx,Col:Idx> SparseBits<Row,Col>{
	pub fn insert(&mut self,row:Row,col:Col){
		assert!(col.index()<self.size);

		let r = &mut self.stroage[row];
		if r.is_none() {
			*r = Some(BS::new(self.size));

		}

		r.as_mut().unwrap().insert(col)
	}

	pub fn merge_into(&mut self,src:Row,dst:Row){
		if src==dst {
			return;
		}

		let (src,dst) = self.stroage.pick2_mut(src,dst);
		
		if dst.is_none() {
			*dst=src.clone();	
		}

		let Some(src) = src else {
			return;
		};

		dst.as_mut().unwrap().merge_with(src);
	}

	pub fn contains(&self,row:Row,col:Col)->bool {
		assert!(col.index()<self.size);

		match &self.stroage[row]{
			None=>false,
			Some(r)=>r.contains(col)
		}
	}

	#[inline(always)]
	pub fn iter_bools(&self,row:Row)->impl Iterator<Item=bool>+ExactSizeIterator{
		let r = &self.stroage[row];
		(0..self.size).map(move |i|{
			match r {
				Some(bs) => bs.contains(Col::new(i)),
				None=>false
			}
		})
	}

	pub fn iter(&self,row:Row)->impl Iterator<Item=Col>{
		self.iter_bools(row).enumerate().filter_map(|(i,b)| {
			if b {
				Some(Col::new(i))
			}else{
				None
			}
		})
	}
}

pub fn propegate_constraints<Node:Idx,Other:Idx>(sets:&mut SparseBits<CId<Node>,Other>,sccs:&SCCS<Node>){
	for c in sccs.o_dag.iter_nodes(){
		for child in sccs.o_dag.edges(c){
			if c==child { continue };
			sets.merge_into(child,c);
		}
	}	
}