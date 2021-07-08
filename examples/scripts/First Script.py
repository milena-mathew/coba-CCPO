"""
This is an example script that creates a Benchmark.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, ChanceConstrainedOptimizer
from coba.simulations import ValidationSimulation
from coba.benchmarks import Benchmark

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    #First, we define the learners that we want to test
    learners = [
        VowpalLearner(bag=5, seed=10),      #This learner requires that VowpalWabbit be installed
        ChanceConstrainedOptimizer(constraint=0.1, learning_rate=0.3, vw_kwargs={"bag":5, "seed":10})
    ]

    #Then we define the simulations that we want to test our learners on
    simulations = [ ValidationSimulation(300, context_features=True, action_features=False, seed=1000) ]

    #And also define a collection of seeds used to shuffle our simulations
    seeds = [0,1,2,3]

    #We then create our benchmark using our simulations and seeds
    benchmark = Benchmark(simulations, shuffle=seeds)

    #Finally we evaluate our learners on our benchmark (the results will be saved in `result_file`).
    result = benchmark.evaluate(learners)

    #After evaluating can create a quick summary plot to get a sense of how the learners performed
    result.plot_learners()

    #We can also create a plot examining how one specific learner did across each shuffle of our simulation
    result.plot_shuffles(learner_pattern="vw")