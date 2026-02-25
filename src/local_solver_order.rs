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
    use_main_loop_deferred_mode: bool,
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

        if self.use_main_loop_deferred_mode {
            order.push(LocalSolverPass::Deferred);
        }

        order.shuffle(&mut self.rng);

        order
    }

    #[inline(always)]
    pub(crate) fn use_main_loop_deferred_mode(&self) -> bool {
        self.use_main_loop_deferred_mode
    }

    fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let use_main_loop_deferred_mode = rng.random();

        eprintln!(
            "[solver-order-fuzz] seed={seed} use_main_loop_deferred_mode={use_main_loop_deferred_mode}"
        );

        Self {
            rng,
            use_main_loop_deferred_mode,
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

        assert_eq!(a.use_main_loop_deferred_mode, b.use_main_loop_deferred_mode);
    }
}
