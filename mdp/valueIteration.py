from utils import util

class ValueIteration:
    """
        A ValueIteration algorithm takes a Markov decision process
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

    def runValueIteration(self):
        iterCount = 0
        oldValues = util.Counter()
        while True:
            # pre-store the state values of last iteration
            for state,value in self.values.items():
                oldValues[state] = value

            delta = 0.0
            for state in self.mdp.getStates():
                actions = self.mdp.getPossibleActions(state)
                if len(actions) == 0:
                    continue

                actionValues = util.Counter()
                for action in actions:
                    successors = self.mdp.getTransitionStatesAndProbs(state, action)
                    saValue = 0
                    for nextState,prob in successors:
                        saValue += prob * (self.mdp.getReward(state, action, nextState) +
                                           self.gamma * oldValues[nextState])
                    actionValues[action] = saValue
                maxValue = actionValues[actionValues.argMax()]
                self.values[state] = maxValue
                delta = max(delta, abs(self.values[state] - oldValues[state]))

            iterCount += 1
            #if iterCount % 100 == 0:
            print("Iteration: ", iterCount)
            if iterCount >= self.iterations or delta <= self.theta:
                break
        print("Value Iteration Converged!")
        print("delta: ", delta)



    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        pass

    def computeActionFromValues(self, state):
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:   # terminal state
            return None

        actionValues = util.Counter()
        for action in actions:
            successors = self.mdp.getTransitionStatesAndProbs(state, action)
            saValue = 0.0
            for nextState, prob in successors:
                saValue += prob * (self.mdp.getReward(state, action, nextState) +
                                   self.gamma * self.values[nextState])
            actionValues[action] = saValue
        #maxValue = actionValues[actionValues.argMax()]
        #bestActions = [action for action,value in actionValues.items() if value == maxValue]
        #return random.choice(bestActions)   # break ties if any
        bestAction = actionValues.argMax()  # don't care about ties
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        pass




