from Environment import game
from model import DQN, AItrainer
import numpy as np
from collections import deque
import random
from helper import plot
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import pygame
from keras.models import load_model
MAX_MEMORY = 100000
BATCH_SIZE = 1000
random.seed(1000)

class agent():
    def __init__(self):
        self.n_games = 0
        self.e_greedy = 1000 # randomness
        self.gamma = 0.9 # discount rate
        self.learning_rate = 0.001
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.structure = DQN()
        self.train_method = AItrainer(self.structure,self.learning_rate,self.gamma)
        
    def choose_action(self,old_state,steps):
        # self.e_greedy = 500 - self.n_games
        self.e_greedy = 50 - self.n_games
        action = [0,0,0]
        if np.random.uniform(low = 0, high = 600) < self.e_greedy or steps >50:
            move =random.randint(0,2)
            action[move] = 1
            temp = 0
        else:
            old_state = tf.convert_to_tensor(old_state)
            old_state = tf.expand_dims(old_state,axis = 0)
            prediction = self.structure.predict(old_state)
            index = np.argmax(prediction[0],axis =0)
            action[index] = 1
            temp = 1

        return action,temp

    def remember(self,state,action,reward,next_state,done,_):
        self.memory.append((state,action,reward,next_state,done,_))

    def train_short_long_memory(self,Time,state = 0,action = 0,reward = 0,next_state = 0,done = 0,_ = 0):
        if Time:
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            else:
                mini_sample = self.memory
            state,action,reward,next_state,done,_ = zip(*mini_sample)
            state = list(state)
            action = list(action)
            reward = list(reward)
            next_state = list(next_state)
            done = list(done)
            _ = list(_)
        self.train_method.train_step(state,action,reward,next_state,done,_)


def AI_SNAKE():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 10

    env = game()
    Agent = agent()
    optimizer = Adam(Agent.learning_rate)
    loss = MeanSquaredError()
    Agent.train_method.compile(optimizer,loss)
    Agent.structure.load_weights('./MyAIsnake')
    # Agent.train_method.structure = load_model('C:\\Users\\User\\Desktop\\python game\\SnakeAI\\my_model.h5')
    observation = env.get_state()
    for episode in range(3000):
        # initial observation
        while True:
            # fresh env
            for event in pygame.event.get():
                pass
            env.draw()
            # print("fine")
            action,_ = Agent.choose_action(observation,env.move)
            # print("ok")
            # RL take action and get next observation and reward
            observation_, reward, running = env.update(action)
            # print(observation_,running,action)
            Agent.train_short_long_memory(False,observation, action, reward,observation_, running,_)
            Agent.remember(observation, action, reward,observation_, running,_)
            observation = observation_
            # break while loop when end of this episode
            if not running:
                Agent.n_games +=1
                # print("great")
                Agent.train_short_long_memory(True)
                # if env.score > record:
                #     record = score
                #     Agent.structure.save("mymodel.h5")

                # if env.score >=100:
                #     Agent.structure.save_weights('./MyAIsnake')
                #     quit()
                plot_scores.append(env.score)
                total_score += env.score
                mean_score = total_score / Agent.n_games
                plot_mean_scores.append(mean_score)
                print('Game', Agent.n_games, 'Score', env.score, 'Record:', record)
                plot(plot_scores, plot_mean_scores)
                break
        observation = env.reset()

    # end of game
    print('game over')
    env.quit()

# def run_snack():
#     env = game()
#     Agent = agent()
#     optimizer = Adam(Agent.learning_rate)
#     loss = MeanSquaredError()
#     Agent.train_method.compile(optimizer,loss)



if __name__ == '__main__':
    AI_SNAKE()