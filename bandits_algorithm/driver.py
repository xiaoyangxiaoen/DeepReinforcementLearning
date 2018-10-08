import gym
import gym_bandits
from bandits_algorithm import banditsAlgorithm
from matplotlib import pyplot as plt


def runGreedyBanditsAlgorithm(
        banditEnvId="BanditTenArmedGaussian-v0",    # bandit id
        numArms=10,                                 # number of bandit arms
        totalSteps=1000,                            # number of experience steps
        totalRuns=2000):                            # number of bandit task

    avgRewards = [0.0] * totalSteps
    numRuns = 0
    while numRuns < totalRuns:
        banditEnv = gym.make(banditEnvId)
        banditEnv.reset()
        gba = banditsAlgorithm.GreedyBanditsAlgorithm(banditEnv, numArms=numArms)
        numSteps = 0
        while numSteps < totalSteps:
            gba.update()
            numSteps += 1
        avgRewards = [avgRewards[i] + gba.rewardPerStep[i] for i in range(totalSteps)]
        numRuns += 1
        if numRuns % 100 == 0:
            print("Runs: ",numRuns)
    avgRewards = [avgRewards[i] / len(avgRewards) for i in range(len(avgRewards))]
    return avgRewards


def runBanditsAlgorithm(
        algoFunc,                                   # a specific bandit algorithm function
        *args,                                      # positional parameters of algoFunc
        banditEnvId="BanditTenArmedGaussian-v0",    # bandit id
        totalSteps=1000,                            # number of experience steps
        totalRuns=2000,                             # number of bandit task
        **kwargs):                                  # keyword parameters of algoFunc

    avgRewards = [0.0] * totalSteps
    numRuns = 0
    while numRuns < totalRuns:
        banditEnv = gym.make(banditEnvId)
        banditEnv.reset()
        gba = algoFunc(banditEnv, *args, **kwargs)
        numSteps = 0
        while numSteps < totalSteps:
            gba.update()
            numSteps += 1
        avgRewards = [avgRewards[i] + gba.rewardPerStep[i] for i in range(totalSteps)]
        numRuns += 1
        if numRuns % 200 == 0:
            print("Runs: ", numRuns)
    avgRewards = [avgRewards[i] / len(avgRewards) for i in range(len(avgRewards))]
    return avgRewards


if __name__ == '__main__':

    avgRewardsUpperConfidenceBound = runBanditsAlgorithm(
        banditsAlgorithm.UpperConfidenceBoundBanditsAlgorithm,
        banditEnvId="BanditTenArmedGaussian-v0",
        totalSteps=1000,
        totalRuns=2000,
        numArms=10,
        C = 2.0)

    avgRewardsGreedy = runBanditsAlgorithm(
        banditsAlgorithm.GreedyBanditsAlgorithm,
        banditEnvId="BanditTenArmedGaussian-v0",
        totalSteps=1000,
        totalRuns=2000,
        numArms=10)

    avgRewardsEpsilonGreedy = runBanditsAlgorithm(
        banditsAlgorithm.EpsilonGreedyBanditsAlgorithm,
        banditEnvId="BanditTenArmedGaussian-v0",
        totalSteps=1000,
        totalRuns=2000,
        numArms=10,
        epsilon=0.1)




    # plot rewards with time step:
    timeSteps = [i + 1 for i in range(len(avgRewardsGreedy))]
    plt.ylabel("AvgRewards")
    plt.xlabel("TimeSteps")
    plt.title("Average Rewards vs. Time Steps")
    plt.tight_layout()
    plt.plot(timeSteps, avgRewardsGreedy, label="Greedy")
    plt.plot(timeSteps, avgRewardsEpsilonGreedy, label="EpsilonGreedy")
    plt.plot(timeSteps, avgRewardsUpperConfidenceBound, label="UpperConfidenceBound")
    plt.legend()
    plt.show()