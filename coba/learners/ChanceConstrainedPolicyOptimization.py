"""The expected interface for all learner implementations."""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Dict

from coba.simulations import Context, Action, Key
from coba.random import CobaRandom
from coba.utilities import DiagonalFreeGrad
from coba.learners.core import Learner


class ChanceConstrainedOptimizer(Learner):
    """The interface for Learner implementations."""

    @property
    def family(self) -> str:
        """The family of the learner.

        This value is used for descriptive purposes only when creating benchmark results.
        """
        return self._learner.family + "-chanceconstrained"

    @property
    def params(self) -> Dict[str,Any]:
        """The parameters used to initialize the learner.

        This value is used for descriptive purposes only when creating benchmark results.
        """
        return {"vw_params":self._learner.params, "learning_rate_rho": self._rho}

    def __init__(self, constraint, len_feedback, learning_rate: float, learner, vw_args=[], vw_kwargs={}) -> None:
        """An optional initialization method called once after pickling."""        
        self._learner = learner(*vw_args, **vw_kwargs) 
        self._rho = learning_rate
        self._freegrad = DiagonalFreeGrad(d=len_feedback)

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            actions: The current set of actions to choose from in the given context. 
                Action sets can be lists of numbers (e.g., [1,2,3,4]), a list of 
                strings (e.g. ["high", "medium", "low"]), or a list of tuples such 
                as in the case of movie recommendations (e.g., [("action", "oscar"), 
                ("fantasy", "razzie")]).

        Returns:
            A sequence of probabilities indicating the probability for each action.
        """
        return self._learner.predict(key, context, actions)

    def learn(self, key: Key, context: Context, action: Action, feedback: Any, probability: float) -> None:
        """Learn about the result of an action that was taken in a context.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            action: The action that was selected to play and observe its reward. 
                An Action can be an individual number (e.g., 2), a string (e.g. 
                "medium"), or a list of some combination of numbers or strings
                (e.g., ["action", "oscar"]).
            feedback: The feedback received for taking the given action in the given context.
            probability: The probability with wich the given action was selected.
        """
        
        def grad(*args):
            return [f for f in feedback[1:]]

        duals = self._freegrad.update(gradientfn=grad)
        duals = [min(v, 0) for v in duals]
        adjusted_reward = feedback[0] + sum(a * b for (a, b) in zip(duals, feedback[1:])) 
        return self._learner.learn(key, context, action, adjusted_reward, probability)
    
