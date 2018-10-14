from utils import util
import random
import math

class BanditsAlgorithm:

    def __init__(self, banditEnv, numArms=10):
        self.Q = util.Counter()         # action values
        self.N = util.Counter()         # number of occurences of action
        self.numArms = numArms          # number of arms (actions), start from 0, e.g. {0,1,2,3,4} for five arms
        self.env = banditEnv            # multiple-armed bandits
        self.rewardPerStep = []         # reward per time step
        self.numSteps = 0               # current number of steps

        for i in range(self.numArms):   # explicitly initialize Q and N
            self.Q[i] = 0.0
            self.N[i] = 0.0

    def chooseAction(self):         # must be overide
        """
        choose the best action for current time step
        :return:
        """
        util.raiseNotDefined()

    def doAction(self, action):
        """
        do action to environment to get an immediate reward
        :param action:
        :return:
        """
        _, reward, _, _ = self.env.step(action)
        return reward

    def update(self):
        """
        update Q and N
        :return:
        """
        bestAction = self.chooseAction()
        reward = self.doAction(bestAction)
        self.N[bestAction] += 1
        self.Q[bestAction] = self.Q[bestAction] + (reward - self.Q[bestAction]) / self.N[bestAction]
        self.rewardPerStep.append(reward)
        self.numSteps += 1


class GreedyBanditsAlgorithm(BanditsAlgorithm):

    def __init__(self, banditEnv, **kwargs):
        BanditsAlgorithm.__init__(self, banditEnv, **kwargs)

    def chooseAction(self):
        """
        choose the best action for current time step
        :return:
        """
        if len(self.Q.keys()) == 0:
            return None
        maxValue = self.Q[self.Q.argMax()]
        bestActions = [key for key,value in self.Q.items() if value == maxValue]    # break ties if any
        return random.choice(bestActions)


class EpsilonGreedyBanditsAlgorithm(BanditsAlgorithm):

    def __init__(self, banditEnv, epsilon=0.01, **kwargs):
        BanditsAlgorithm.__init__(self, banditEnv, **kwargs)
        self.epsilon = epsilon

    def chooseAction(self):
        """
        choose the best action for current time step
        :return:
        """
        if len(self.Q.keys()) == 0:
            return None
        bestActions = [i for i in range(self.numArms)]

        # a random action with probability epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(bestActions)

        # best action argmaxQ(a) with probability 1-epsilon
        maxValue = self.Q[self.Q.argMax()]
        bestActions = [key for key,value in self.Q.items() if value == maxValue]    # break ties if any
        return random.choice(bestActions)


class UpperConfidenceBoundBanditsAlgorithm(BanditsAlgorithm):

    def __init__(self, banditEnv, C=2.0, **kwargs):
        BanditsAlgorithm.__init__(self, banditEnv, **kwargs)
        self.C = C

    def chooseAction(self):
        ucbValues = []
        for i in range(self.numArms):
            ucb = None
            if self.N[i] != 0:
                ucb = self.Q[i] + self.C * math.sqrt(math.log(self.numSteps) / self.N[i])
            else:
                ucb = float('inf')
            ucbValues.append(ucb)
        maxValue = max(ucbValues)
        bestActions = [i for i in range(self.numArms) if maxValue == ucbValues[i]]
        return random.choice(bestActions)



# gradient-bandit, to be continued

# optimistic-initial-value, to be continued