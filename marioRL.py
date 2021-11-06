#Import Dependencies
import tensorflow as tf
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Activation,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from IPython.display import clear_output
from keras.models import save_model,load_model
import time
from PIL import Image

#Setting up gym environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)
total_reward = 0
done = True

# for step in range(100000):
#     env.render()
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step(env.action_space.sample())
#     print(info)
#     total_reward += reward
#     clear_output(wait=True)

# env.close()

#Building class for Mario Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        #Variables for agent
        self.state_space = state_size
        self.action_space = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.8
        self.chosenAction = 0

        #Exploration vs Exploitation
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_epsilon = 0.0001

        #Building Neural Network for Agent
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()


    def build_network(self):
        model= Sequential()
        model.add(Conv2D(64,(4,4), strides=4, padding="same",input_shape= self.state_space))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(4,4), strides=2, padding="same"))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(4,4), strides=1, padding="same"))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(512,activation='relu'))
        model.add(Dense(256,activation='relu'))
        model.add(Dense(self.action_space,activation='linear'))

        model.compile(loss='mse', optimizer=Adam())
        return model   

    #Update target network function
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    #Action function
    def act(self,state,onGround):
        if onGround < 83:
            print("On Ground")
            if random.uniform(0,1) < self.epsilon:
                self.chosenAction = np.random.randint(self.action_space)
                return self.chosenAction

            Q_value = self.main_network.predict(state)
            self.chosenAction = np.argmax(Q_value[0])
            # print(Q_value)
            return self.chosenAction
        else:
            print("Not on Ground")
            return self.chosenAction

    #Updating epsilon fucntion
    def update_epsilon(self,episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_epsilon * episode)

    #Train function
    def train(self, batch_size):
        #Mini Batch from Memory
        mini_batch = random.sample(self.memory, batch_size)

        #Getting variables to calculate Q-Value
        for state, action, reward, next_state, done in mini_batch:
            target = self.main_network.predict(state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = (reward + self.gamma * np.amax(self.target_network.predict(next_state)))

            self.main_network.fit(state, target, epochs=1, verbose=0)

    def store_transition(self,state,action, reward,next_state,done):
        self.memory.append((state, action, reward, next_state, done))

    def get_pred_act(self,state):
        Q_values = self.main_network.predict(state)
        return np.argmax(Q_values[0])

    def load(self,name):
        self.main_network = load_model(name)
        self.target_network = load_model(name)

    def save(self,name):
        save_model(self.main_network,name)

#Pre-process state for Neural Network Function
action_space = env.action_space.n
state_space = (80,88,1)

def preprocess_state(state):
    image = Image.fromarray(state)
    image = image.resize((88,80))
    image = image.convert('L')
    image = np.array(image)

    return image

# env.observation_space
# dqnAgent = DQNAgent(state_size)

#Learn Function
num_episodes = 1000000
num_timesteps = 400000
batch_size = 64
DEBUG_LENGTH = 300

dqn = DQNAgent(state_space,action_space)
print("Starting Training")

stuck_buffer = deque(maxlen=DEBUG_LENGTH)

for i in range(num_episodes):
    Return = 0
    done = False
    time_step = 0
    onGround = 79
    state = preprocess_state(env.reset())
    state = state.reshape(-1,80,88,1)

    for t in range(num_timesteps):
        env.render()
        time_step += 1
        if t>1 and stuck_buffer.count(stuck_buffer[-1]) > DEBUG_LENGTH -50:
            action = dqn.act(state,onGround=79)
        else:
            action = dqn.act(state,onGround)

        print("Action: " + str(action))

        next_state, reward, done, info = env.step(action)
        # print(info)
        onGround = info["y_pos"]
        stuck_buffer.append(info["x_pos"])
        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1,80,88,1)

        dqn.store_transition(state,action,reward,next_state,done)
        state = next_state

        Return += reward
        print("Episode is: {}\nTotal Time Step: {}\nCurrent Reward: {}\nEpsilon is: {}".format(str(i),str(time_step),str(Return),str(dqn.epsilon)))
        clear_output(wait=True)

        if done:
            break

        if len(dqn.memory) > batch_size and i > 5:
            dqn.train(batch_size)

    dqn.update_epsilon(i)
    clear_output(wait=True)
    dqn.update_target_network()
    #save model
    dqn.save('marioRL.h5')

env.close()

dqn.load('marioRL.h5')

#Visualizing model
while 1:
    done = False
    state = preprocess_state(env.reset())
    state = state.reshape(-1,80,88,1)
    total_reward = 0

    while not done:
        env.render()
        action = dqn.get_pred_act(state)
        next_state,reward,done,info = env.step(action)
        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1,80,88,1)
        state = next_state

env.close()