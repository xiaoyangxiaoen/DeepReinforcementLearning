from mdp import gamblerMDP
from mdp import valueIteration
from mdp import policyIteration
from matplotlib import pyplot as plt


def runValueIterationForGamblerMDP(
        pCoinHead = 0.25,        # probability of coins head up
        goal = 100,              # money goal of the gambler
        gamma = 1.0,             # discount factor of markov decision process
        iterations = 100,        # max iterations of value iteration
        theta = 0.0001):         # convergence threshold of value iteration

    mdp = gamblerMDP.GamblerMDP(pCoinHead, goal)
    viAlgo = valueIteration.ValueIteration(
        mdp,                        # markov decision process to be solved
        gamma=gamma,                # discount factor
        iterations=iterations,      # max iterations
        theta=theta)                # convergence threshold

    print("Start Value Iteration...")
    viAlgo.runValueIteration()

    print("Extract Policy...")
    policy = [viAlgo.getPolicy(state) for state in range(1, goal)]
    values = [viAlgo.getValue(state) for state in range(1, goal)]

    # plot stake vs time capital:
    capital = [i for i in range(1, goal)]
    plt.subplot(2, 1, 1)
    plt.ylabel("Value estimates")
    plt.title("Value Iteration")
    plt.plot(capital, values)

    plt.subplot(2, 1, 2)
    plt.ylabel("Final policy (stake)")
    plt.xlabel("Capital")
    plt.bar(capital, policy, align='center', alpha=0.5)
    plt.legend()
    plt.show()


def runPolicyIterationForGamblerMDP(
        pCoinHead = 0.25,           # probability of coins head up
        goal = 100,                 # money goal of the gambler
        gamma = 1.0,                # discount factor of markov decision process
        iterations = 100,           # max iterations of policy iteration
        theta = 0.0001):            # convergence threshold of policy evaluation

    mdp = gamblerMDP.GamblerMDP(pCoinHead, goal)
    piAlgo = policyIteration.PolicyIteration(
        mdp,                        # markov decision process to be solved
        gamma=gamma,                # discount factor
        iterations=iterations,      # max iterations
        theta=theta)                # convergence threshold of policy evaluation

    print("Start Policy Iteration...")
    piAlgo.runPolicyIteration()

    print("Extract Policy...")
    policy = [piAlgo.getPolicy(state) for state in range(1, goal)]
    values = [piAlgo.getValue(state) for state in range(1, goal)]

    # plot stake vs time capital:
    capital = [i for i in range(1, goal)]
    plt.subplot(2, 1, 1)
    plt.ylabel("Value estimates")
    plt.title("Policy Iteration")
    plt.plot(capital, values)

    plt.subplot(2, 1, 2)
    plt.ylabel("Final policy (stake)")
    plt.xlabel("Capital")
    plt.bar(capital, policy, align='center', alpha=0.5)
    plt.legend()
    plt.show()

if __name__ == '__main__':

    # value iteration for Gambler markov decision process
    runValueIterationForGamblerMDP()

    # policy iteration for Gambler markov decision process
    #runPolicyIterationForGamblerMDP()   # result is susceptable to initial policy