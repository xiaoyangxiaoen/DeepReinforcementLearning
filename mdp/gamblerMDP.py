from mdp import baseMDP
import random

class GamblerMDP(baseMDP.MarkovDecisionProcess):
    """
    Gambler Markov Decision Process
    """

    def __init__(self, pCoinHead = 0.4, goal = 100):
        self.pCoinHead = pCoinHead
        self.goal = goal
        self.isTransited = {0: False, self.goal: False}    # whether transited from terminal state

    def getStates(self):
        """
        return list of all states
        """
        return [state for state in range(self.goal + 1)]   # 0 and goal is the terminal state

    def getStartState(self):
        """
        generate a random start state
        """
        return random.choice(self.getStates())

    def getPossibleActions(self, state):
        """
        Returns list of valid actions for 'state'.
        """
        if self.isTerminal(state):
            if not self.isTransited[state]:
                return ['EXIT']
            else:
                return []
        # caution: minimum stake is 1, not 0
        return [action for action in range(1, min(state, self.goal - state) + 1)]

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """
        if action not in self.getPossibleActions(state):
            return []
        if self.isTerminal(state):  # terminal states only transit once
            if not self.isTransited[state]:
                self.isTransited[state] = True
                return [(state, 1.0)]
            else:
                return []

        successors = []
        successors.append((state + action, self.pCoinHead))         # win action $
        successors.append((state - action, 1 - self.pCoinHead))     # loss action $
        return successors


    def getReward(self, state, action, nextState):
        """
        Get reward for state, action, nextState transition.

        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        if state == self.goal:
            return 1.0
        return 0.0

    def isTerminal(self, state):
        if state == 0 or state == self.goal:
            return True
        else:
            return False


