use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub(crate) enum LocalSolverPass {
    Operators,
    Deferred,
    PendingIndexes,
    PendingMemberAccesses,
    PendingIntAccesses,
    PendingSpecializations,
    PendingDerefs,
}

pub(crate) struct SolverOrderPlanner {
    rng: StdRng,
    include_deferred_in_main_loop: bool,
    deferred_on_stall: bool,
    use_iterative_deferred_finalize: bool,
}

impl SolverOrderPlanner {
    pub(crate) fn new() -> Self {
        let seed = read_seed_from_env();
        Self::from_seed(seed)
    }

    pub(crate) fn primary_pass_order(&mut self) -> Vec<LocalSolverPass> {
        let mut order = vec![
            LocalSolverPass::Operators,
            LocalSolverPass::PendingIndexes,
            LocalSolverPass::PendingMemberAccesses,
            LocalSolverPass::PendingIntAccesses,
            LocalSolverPass::PendingSpecializations,
            LocalSolverPass::PendingDerefs,
        ];

        if self.include_deferred_in_main_loop {
            order.push(LocalSolverPass::Deferred);
        }

        order.shuffle(&mut self.rng);

        order
    }

    #[inline(always)]
    pub(crate) fn resolve_deferred_on_stall(&self) -> bool {
        self.deferred_on_stall
    }

    #[inline(always)]
    pub(crate) fn use_iterative_deferred_finalize(&self) -> bool {
        self.use_iterative_deferred_finalize
    }

    fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let include_deferred_in_main_loop = rng.random();
        let deferred_on_stall = rng.random();
        let use_iterative_deferred_finalize = rng.random();

        eprintln!(
            "[solver-order-fuzz] seed={seed} include_deferred_in_main_loop={include_deferred_in_main_loop} deferred_on_stall={deferred_on_stall} use_iterative_deferred_finalize={use_iterative_deferred_finalize}"
        );

        Self {
            rng,
            include_deferred_in_main_loop,
            deferred_on_stall,
            use_iterative_deferred_finalize,
        }
    }
}

fn read_seed_from_env() -> u64 {
    match std::env::var("EXPR_SOLVER_ORDER_SEED") {
        Ok(raw) => raw.parse().unwrap_or_else(|_| {
            panic!(
                "failed to parse EXPR_SOLVER_ORDER_SEED as u64: {raw:?}; example: EXPR_SOLVER_ORDER_SEED=123"
            )
        }),
        Err(_) => rand::random(),
    }
}

#[cfg(test)]
mod tests {
    use super::SolverOrderPlanner;

    #[test]
    fn seeded_strategy_is_deterministic() {
        let a = SolverOrderPlanner::from_seed(123456789);
        let b = SolverOrderPlanner::from_seed(123456789);

        assert_eq!(
            a.include_deferred_in_main_loop,
            b.include_deferred_in_main_loop
        );
        assert_eq!(a.deferred_on_stall, b.deferred_on_stall);
        assert_eq!(
            a.use_iterative_deferred_finalize,
            b.use_iterative_deferred_finalize
        );
    }
}
