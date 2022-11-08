# %% codecell
import numpy as np
import gym
import seaborn as sb
import itertools
import sys
import copy
import random

np.seterr(invalid='raise')
# %% codecell
class ValueEstimator:
  def __init__(self, layers, action_space):
    """

    """
    self.action_space = action_space

    self.layers = []

    for i in range(len(layers) - 1):
      weights = np.random.normal(scale=np.sqrt(2/layers[i]), size=(layers[i+1], layers[i]))
      biases = np.zeros((layers[i+1], 1))
      self.layers.append((weights, biases, 'leaky relu'))

    #output layer
    weights = np.random.normal(scale=np.sqrt(2/layers[-1]), size=(1, layers[-1]))
    biases = np.zeros((1, 1))
    self.layers.append((weights, biases, 'linear'))


  def fit(self, states, target, learning_rate=.01, return_grads=False, learning=True):
      """
      """
      pred, caches = self.predict(states)
      dA = MSE_grad(target, pred)
      #dA_norm = np.linalg.norm(dA)
      #dA = dA / dA_norm if (dA_norm>1) else dA

      m = states.shape[1]

      grads = []
      for i in reversed(range(len(self.layers))):
          A_prev, W, b, Z = caches[i]

          #Activation function derivatives
          if self.layers[i][2] == 'relu':
              dZ = dA
              dZ[Z <= 0] = 0
          elif self.layers[i][2] == 'leaky relu':
              dZ = np.where(Z > 0, dA, dA * 0.01)
          elif self.layers[i][2] == 'linear':
              dZ = dA
          elif self.layers[i][2] == 'sigmoid':
              raise NotImplementedError('Sigmoid derivitive not implemented.')

          #Weight function derivatives
          dW = np.dot(dZ, A_prev.T)/m
          dB = np.sum(dZ, axis = 1, keepdims=True)/m
          dA = np.dot(self.layers[i][0].T, dZ)

          """grads.append(np.sum(dB))
          grads.append(np.sum(dW))"""


          #Gradient Updates
          #print(f"""Weight grad: {-(learning_rate*dW)}
          #Bias grad: {-(learning_rate*dB)}""")
          if learning:
              self.layers[i] = (self.layers[i][0]-(learning_rate*dW), self.layers[i][1]-(learning_rate*dB), self.layers[i][2])

      if return_grads:
          return dA, np.flip(np.array(grads))
      return dA


  def predict(self, X, grad_check_epsilon=0.0, grad_check_param=None, print_layer_outputs=False):
    """
    Forward-propogates an input through the model to create a prediction on the value of each action.

    Args:
    X - Array of all inputs
    """
    caches = []
    if (grad_check_param == None):
        for i, l in enumerate(self.layers):
            A = X.copy()
            X = np.dot(l[0], X) + l[1]
            cache = (A, l[0].copy(), l[1].copy(), X.copy())
            caches.append(cache)

            if l[2] == 'relu':
                np.maximum(X, 0, out=X)
            elif l[2] == 'leaky relu':
                X = np.where(X > 0, X, X * 0.01)
            elif l[2] == 'linear':
                pass
            elif l[2] == 'sigmoid':
                X = 1/(1+np.exp(-X))

            if print_layer_outputs:
                print(f"Layer {i}", X)
        return X, caches

    #grad check version
    i=0
    for l in self.layers:
        W = l[0] if (i!=grad_check_param) else l[0]+grad_check_epsilon
        b = l[1] if (i+1!=grad_check_param) else l[1]+grad_check_epsilon

        A = X.copy()
        X = np.dot(W, X) + b
        cache = (A, l[0].copy(), l[1].copy(), X.copy())
        caches.append(cache)

        if l[2] == 'relu':
            np.maximum(X, 0, out=X)
        elif l[2] == 'leaky relu':
            X = np.where(X > 0, X, X * 0.01)
        elif l[2] == 'linear':
            pass
        elif l[2] == 'sigmoid':
            X = 1/(1+np.exp(-X))

        i+=2

    return X


  def grad_check(self, X, Y, epsilon=1e-7):
      estimates = []
      i=0
      for l in enumerate(self.layers):
          J_plus_W = float(mean_squared_error(Y, self.predict(X, grad_check_epsilon=epsilon, grad_check_param=i), ax=None))
          J_minus_W = float(mean_squared_error(Y, self.predict(X, grad_check_epsilon=-epsilon, grad_check_param=i), ax=None))
          J_W = (J_plus_W - J_minus_W) / (2*epsilon)
          estimates.append(J_W)

          J_plus_b = float(mean_squared_error(Y, self.predict(X, grad_check_epsilon=epsilon, grad_check_param=i+1), ax=None))
          J_minus_b = float(mean_squared_error(Y, self.predict(X, grad_check_epsilon=-epsilon, grad_check_param=i+1), ax=None))
          J_b = (J_plus_b - J_minus_b) / (2*epsilon)
          estimates.append(J_b)

          i+=2
      estimates = np.array(estimates)

      _, grads = self.fit(X, Y, return_grads=True, learning=False)

      numerator = np.linalg.norm(grads - estimates)
      denominator = np.linalg.norm(grads) + np.linalg.norm(estimates)
      difference = numerator / denominator

      return difference, estimates, grads


  def addActionstoState(self, states, a=None):
      if a == None:
          actions = np.zeros((self.action_space, self.action_space))
          for i in range(self.action_space):
              actions[i, i] = 1
          actions = np.tile(actions, states.shape[1])

          states = np.repeat(states, self.action_space, axis=1)

      else:
          actions = np.zeros((self.action_space, states.shape[1]))
          actions[a, :] = 1


      #print(states)
      #print(actions)
      states = np.concatenate((states, actions), axis=0)
      return states


def mean_squared_error(target, predicted, ax=0):
    #print(target, "|", predicted)
    return np.mean(np.square(target - predicted), axis=ax, keepdims=True)

def MSE_grad(target, predicted):
    try:
        return (-2*np.sum((target - predicted), axis=0, keepdims=True))/target.shape[0]
    except FloatingPointError:
        print(target, predicted)
        sys.exit()

# %% codecell
estimator = ValueEstimator((4,5,5,5,5), 2)
next_state = np.array([[1, 0, 2],
                       [1, 0, 4]])
aaa = estimator.predict(estimator.addActionstoState(next_state))[0]
print(aaa)
print()
np.reshape(aaa, (2, -1), 'F')

# %% codecell
estimator = ValueEstimator((1,5,5,5,5), 1)
#print(estimator.layers)
test = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 1]])
target = np.array([[1, 2, 3, 4]])
nums = np.array([np.random.uniform(-100, 100, 100)])

for i in range(100000):
    nums = np.array([np.random.uniform(-1000, 1000, 100)])
    estimator.fit(nums, -nums, learning_rate=.001)
    """if (i==49999):
        print(estimator.fit(nums, -nums, learning_rate=.01))"""
    if (i%1000==0):
        demo, _ = estimator.predict(nums)
        print(mean_squared_error(-nums, demo, ax=None))

"""for i in range(10000):
    nums = np.array([np.random.uniform(0, 100, 100)])
    estimator.fit(test, target, learning_rate=.01)
    demo, _ = estimator.predict(test)
    if (i%1000==0):
        print(demo)"""
# %% codecell
#estimator = ValueEstimator((1,3,3), 1)
"""test = np.array([[0],
                  [0]])
target = np.array([[1]])"""
demo, _ = estimator.predict([[0.4]], print_layer_outputs=False)
print(demo)
#estimator.grad_check(test, target)
# %% codecell
a, _ = estimator.predict(np.array([[6]]))
print(a)
# %% codecell
class ReplayMemory:
    #Implented primarily by MiniQuark on stackoverflow
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size, combined=True):
        indices = random.sample(range(self.size), batch_size)
        if combined:
            indices.append(self.index-1)
        return [self.buffer[index] for index in indices]


def epsilon_greedy_policy(observation, estimator, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(estimator.addActionstoState(observation[:, np.newaxis]))[0].flatten()
    best_action = []
    for i, action in enumerate(q_values):
                if action == max(q_values):
                    best_action.append(i)

    for a in best_action:
        A[a] += (1.0 - epsilon)/len(best_action)
    return A


def reinforce(env, estimator, num_episodes, discount_factor=.99, start_epsilon=1.0, end_epsilon = 0.1, epsilon_steps=1000000, learning_rate = .00025, minibatch_size = 32, replay_memory_size = 1000000, target_update_freq = 100000, replay_memory = None):

    #Initialize persistent variables
    total_reward = 0
    error_sum = 0
    i_step = 0
    epsilon_step = (start_epsilon - end_epsilon) / epsilon_steps
    replay_memory = ReplayMemory(replay_memory_size) if (replay_memory == None) else replay_memory

    #Create target estimator
    target_estimator = copy.deepcopy(estimator)

    #Fills replay memory
    if replay_memory.size < minibatch_size:
        state = env.reset()
        for i in range(minibatch_size - replay_memory.size):
            epsilon = end_epsilon + (epsilon_step*max(epsilon_steps - i_step, 0))
            action_probs = epsilon_greedy_policy(state, estimator, epsilon, env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, done, _ = env.step(action)
            state = estimator.addActionstoState(state[:, np.newaxis], action)

            replay_memory.append((state, next_state, reward, done))

            state = next_state
            if done:
                state = env.reset()
            i_step+=1

    #Start iterating through episodes
    for i_episode in range(num_episodes):

        if (i_episode%50==1):
            print(f"Episode {i_episode-1}, Reward: --{total_reward}--, Error: --{error_sum}--, Steps: {i_step}, Epsilon:{epsilon}")
            print()
        sys.stdout.flush()

        total_reward = 0
        error_sum = 0

        state = env.reset()


        for t in itertools.count():
            #TAKES A NEW STEP
            #samples actions from policy
            epsilon = end_epsilon + (epsilon_step*max(epsilon_steps - i_step, 0))
            action_probs = epsilon_greedy_policy(state, estimator, epsilon, env.action_space.n)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            #takes a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward


            state = estimator.addActionstoState(state[:, np.newaxis], action)
            replay_memory.append((state, next_state, reward, done))

            if (i_episode % 50==0):
                q_values_next = estimator.predict(estimator.addActionstoState(next_state[:, np.newaxis]))[0] if not done else 0
                td_target = reward + discount_factor*np.max(q_values_next)

                if np.isnan(td_target):
                    print(td_target)
                    raise OverflowError()

                error_sum += mean_squared_error([[td_target]], estimator.predict(state)[0])
                env.render()

            #TRAINS FROM REPLAY MEMORY
            #samples from replay memory
            states, next_states, rewards, dones  = map(np.array, zip(*replay_memory.sample(minibatch_size)))
            states = states.T
            next_states = next_states.T

            #predicts based on the next states
            try:
                q_values_nexts = np.reshape(estimator.predict(estimator.addActionstoState(next_states))[0], (env.action_space.n, -1), 'F')
            except ValueError:
                print(next_states)
                raise ValueError()

            q_values_nexts = np.where(dones, 0, np.amax(q_values_nexts, axis=0))
            td_targets = rewards + discount_factor*q_values_nexts

            #fits model
            estimator.fit(state, td_targets[:, np.newaxis], learning_rate=learning_rate)

            #updates target estimator to actual
            if (i_step % target_update_freq) == 0:
                print("Target updated!")
                target_estimator.layers = copy.deepcopy(estimator.layers)

            #print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward), end="")

            i_step += 1
            if done:
                if (i_episode % 50==0):
                    print(estimator.predict(state)[0])
                    #print(action_probs)
                    error_sum = (error_sum/t)[0][0]
                if (i_episode==num_episodes-1):
                    pass
                    #print(estimator.grad_check(state, np.array([[td_target]])))
                break

            state=next_state


# %% codecell
env = gym.envs.make("CartPole-v1")
estimator = ValueEstimator((env.observation_space.shape[0]+env.action_space.n, 100, 200, 200, 200, 100), env.action_space.n)


reinforce(env, estimator, 50000, discount_factor=.85, target_update_freq=50000, epsilon_steps=500000, replay_memory_size=200)
# %% codecell
reinforce(env, estimator, 200000, discount_factor=.85, target_update_freq=50000, epsilon_steps=1, replay_memory_size=200)
# %% codecell
print(estimator.layers)
# %% codecell
state = env.reset()
i = 0
while True:
  next_state, reward, done, _ = env.step(1)
  env.render()
  i+=1
  if i%5 == 0:
      print(estimator.predict(estimator.addActionstoState(next_state[:, np.newaxis]))[0])
  if done:
    break
  state=next_state
