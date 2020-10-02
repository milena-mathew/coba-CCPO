"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.simulations import ClassificationSimulation, ShuffleSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import LambdaBatcher, UniversalBenchmark
from coba.analysis import Plots

print("loading simulation data...")
sim = ClassificationSimulation.from_openml(150)

print("shuffling simulation data...")
sim = ShuffleSimulation(sim)

print("defining the benchmark...")
benchmark = UniversalBenchmark([sim], LambdaBatcher(lambda i: 100 + i*100))

print("creating the learners...")
learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.025),
    lambda: UcbTunedLearner(),
    lambda: VowpalLearner(epsilon=0.025),
    lambda: VowpalLearner(bag=5),
    lambda: VowpalLearner(softmax=3.5)
]

print("evaluating the learners...")
results = benchmark.evaluate(learner_factories)

Plots.standard_plot(results)