"""A collection of simple bandit algorithms for comparison purposes."""

import math

from collections import defaultdict
from typing import Any, Dict, Tuple, Sequence, Optional, cast, Hashable

from coba.simulations import Context, Action
from coba.statistics import OnlineVariance
from coba.learners.core import Learner, Key

class RandomLearner(Learner):
    """A Learner implementation that selects an action at random and learns nothing."""

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """  
        return "random"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """
        return {}

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Choose a random action from the action set.
        
        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """
        return [1/len(actions)] * len(actions)

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learns nothing.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """
        pass
 
class EpsilonBanditLearner(Learner):
    """A lookup table bandit learner with epsilon-greedy exploration."""

    def __init__(self, epsilon: float) -> None:
        """Instantiate an EpsilonBanditLearner.

        Args:
            epsilon: A value between 0 and 1. We explore with probability epsilon and exploit otherwise.
            include_context: If true lookups are a function of context-action otherwise they are a function of action.
        """

        self._epsilon = epsilon

        self._N: Dict[Hashable, int            ] = defaultdict(int)
        self._Q: Dict[Hashable, Optional[float]] = defaultdict(int)

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """

        return "bandit_epsilongreedy"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """
        return {"epsilon": self._epsilon }

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        keys        = [ self._key(action) for action in actions ]
        values      = [ self._Q[key] for key in keys ]
        max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
        max_indexes = [i for i in range(len(values)) if values[i]==max_value]

        prob_selected_randomly = [1/len(actions) * self._epsilon] * len(actions)
        prob_selected_greedily = [ int(i in max_indexes)/len(max_indexes) * (1-self._epsilon) for i in range(len(actions))]

        return [p1+p2 for p1,p2 in zip(prob_selected_randomly,prob_selected_greedily)]

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        a_key = self._key(action)
        alpha = 1/(self._N[a_key]+1)

        old_Q = cast(float, 0 if self._Q[a_key] is None else self._Q[a_key])

        self._Q[a_key] = (1-alpha) * old_Q + alpha * reward
        self._N[a_key] = self._N[a_key] + 1

    def _key(self, action: Action) -> Hashable:
        return tuple(action.items()) if isinstance(action,dict) else action

class UcbBanditLearner(Learner):
    """This is an implementation of Auer et al. (2002) UCB1-Tuned algorithm.

    This algorithm assumes that the reward distribution has support in [0,1].

    References:
        Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of 
        the multiarmed bandit problem." Machine learning 47.2-3 (2002): 235-256.
    """
    def __init__(self):
        """Instantiate a UcbBanditLearner."""

        #these variable names were selected for easier comparison with the original paper 
        self._init_a: int = 0
        self._t     : int = 0
        self._s     : Dict[Action, int           ] = defaultdict(int)
        self._m     : Dict[Action, float         ] = {}
        self._v     : Dict[Action, OnlineVariance] = defaultdict(OnlineVariance)

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return "bandit_UCB"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """
        return { }

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        actions = [ self._key(a) for a in actions ]

        #initialize by playing every action once
        if self._init_a < len(actions):
            self._init_a += 1
            return [ int(i == (self._init_a-1)) for i in range(len(actions)) ]

        else:
            values      = [ self._m[a] + self._Avg_R_UCB(a) if a in self._m else None for a in actions ]
            max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
            max_indexes = [i for i in range(len(values)) if values[i]==max_value]

            return [ int(i in max_indexes)/len(max_indexes) for i in range(len(actions)) ]

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        assert 0 <= reward and reward <= 1, "This algorithm assumes that reward has support in [0,1]."

        action = self._key(action)

        if action not in self._m:
            self._m[action] = reward
        else:
            self._m[action] = (1-1/self._s[action]) * self._m[action] + 1/self._s[action] * reward

        self._t         += 1
        self._s[action] += 1
        self._v[action].update(reward)

    def _key(self, action: Action) -> Hashable:
        return tuple(action.items()) if isinstance(action,dict) else action

    def _Avg_R_UCB(self, action: Action) -> float:
        """Produce the estimated upper confidence bound (UCB) for E[R|A].

        Args:
            action: The action for which we want to retrieve UCB for E[R|A].

        Returns:
            The estimated UCB for E[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; n = self._t; n_j = self._s[action]; V_j = self._Var_R_UCB(action)

        return math.sqrt(ln(n)/n_j * min(1/4,V_j))

    def _Var_R_UCB(self, action: Action) -> float:
        """Produce the upper confidence bound (UCB) for Var[R|A].

        Args:
            action: The action for which we want to retrieve UCB for Var[R|A].

        Returns:
            The estimated UCB for Var[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; t = self._t; s = self._s[action]; var = self._v[action].variance

        return var + math.sqrt(2*ln(t)/s)