from utils import util
import random

class PolicyIteration:
    """
        A PolicyIteration algorithm takes a Markov decision process
        (see baseMDP.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, gamma = 1.0, iterations = 500, theta = 0.01):
        self.mdp = mdp                  # markov decision process to be solved
        self.gamma = gamma              # discount factor
        self.iterations = iterations    # max iterations
        self.theta = theta              # a small threshold
        self.values = util.Counter()    # values
        self.policy = util.Counter()    # policy

    def initialize(self):
        for state in self.mdp.getStates():
            self.values[state] = 0.0
            actions = self.mdp.getPossibleActions(state)
            self.policy[state] = random.choice(actions)

    def runPolicyIteration(self):
        # initialize
        self.initialize()

        iterCount = 0
        while True:
            # policy improvement
            oldValues = util.Counter()
            while True:
                # pre-store the state values of last iteration
                for state, value in self.values.items():
                    oldValues[state] = value

                delta = 0.0
                for state in self.mdp.getStates():
                    oldValue = self.values[state]
                    action = self.policy[state]
                    successors = self.mdp.getTransitionStatesAndProbs(state, action)
                    if len(successors) == 0:    # for terminal state if any
                        continue
                    saValue = 0
                    for nextState, prob in successors:
                        saValue += prob * (self.mdp.getReward(state, action, nextState) +
                                           self.gamma * oldValues[nextState])
                    self.values[state] = saValue
                    #delta = max(delta, abs(self.values[state] - oldValues[state]))
                    delta = max(delta, abs(self.values[state] - oldValue))
                if delta <= self.theta:
                    break

            # policy extraction
            policyStable = True
            for state in self.mdp.getStates():
                oldAction = self.policy[state]
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:   # for terminal state, if any
                    self.policy[state] = None
                    continue
                actionValues = util.Counter()
                for action in actions:
                    successors = self.mdp.getTransitionStatesAndProbs(state, action)
                    saValue = 0
                    for nextState, prob in successors:
                        saValue += prob * (self.mdp.getReward(state, action, nextState) +
                                           self.gamma * oldValues[nextState])
                    actionValues[action] = saValue
                bestAction = actionValues.argMax()
                self.policy[state] = bestAction
                if oldAction != self.policy[state]:
                    policyStable = False

            # if policy stable, converge!
            iterCount += 1
            print("Iteration: ", iterCount)
            if policyStable or iterCount >= self.iterations:
                break



    def getValue(self, state):
        return self.values[state]

    def getPolicy(self, state):
        return self.policy[state]

