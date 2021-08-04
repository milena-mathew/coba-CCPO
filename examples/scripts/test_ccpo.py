from typing import Sequence
import coba
from coba.random      import CobaRandom
from coba.simulations import LambdaSimulation, ConstrainedSimulation, Context, Action
from coba.learners    import RandomLearner, VowpalLearner, ChanceConstrainedOptimizer
from coba.benchmarks  import Benchmark

r = CobaRandom()
n_interactions = 1000

def context(index: int) -> Context:
    return tuple(r.randoms(5))

def actions(index: int, context: Context) -> Sequence[Action]:
    actions = [ r.randoms(5) for _ in range(3) ]
    return [ tuple(a/sum(action) for a in action) for action in actions ]

def rewards(index: int, context: Context, action: Action) -> float:
    return sum(c*a for c,a in zip(context,action))

def feedback(index: int, context: Context, action: Action) -> float:
    return tuple((sum(c*a for c,a in zip(context,action)), sum(-c-a for c,a in zip(context,action)), sum(-c-5 for c,a in zip(context,action))))

sim = [ConstrainedSimulation(n_interactions, context, actions, feedback)]

Benchmark(sim).evaluate([ChanceConstrainedOptimizer(constraint=0.1, len_feedback=2, learning_rate=0.3, learner=VowpalLearner, vw_kwargs={"bag":5, "seed":10}), ChanceConstrainedOptimizer(constraint=0.1, len_feedback=2, learning_rate=0.3, learner=RandomLearner)]).plot_learners()

ref_sim = [LambdaSimulation(n_interactions, context, actions, rewards)]

Benchmark(ref_sim).evaluate([VowpalLearner(constraint=0.1, learning_rate=0.3, vw_kwargs={"bag":5, "seed":10})]).plot_learners()
