import numpy as np
import tensorflow as tf
from keras import models
from keras.models import Sequential  
from keras.layers import *
from keras.activations import relu, sigmoid
import time
# import random
np.random.seed(1)
# tf.set_random_seed(1)

def DQN():
    model = Sequential()
    # model.add(Conv2D(32, (3,3), padding='valid', input_shape=(14,14,1)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(32, (3,3), padding='valid'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Flatten())
    model.add(Dense(units=256,activation='relu',input_dim=20))
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=3,activation='softmax'))
    return model


class AItrainer(models.Model):
    def __init__(self,structure,lr,gamma,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.structure = structure
        self.learning_rate = lr
        self.gamma = gamma

    def compile(self,optimizer,loss,*args, **kwargs):
        super().compile(*args, **kwargs)
        self.optimizer = optimizer
        self.loss = loss

    # def loss_call(self, y_real, y_pred):
    #     with tf.GradientTape() as tape:
    #         loss_value = self.loss(y_real,y_pred)
    #     return loss_value
    
    def softmax(self,x):
        e_x = np.exp(x - np.amax(x))
        return e_x / e_x.sum()

    def train_step(self,state,action,reward,next_state,done,_):
        state = tf.convert_to_tensor(state)
        action = tf.convert_to_tensor(action)
        reward = tf.convert_to_tensor(reward)
        next_state = tf.convert_to_tensor(next_state)
        
        if len(state.shape) == 1:
            state = tf.expand_dims(state,axis = 0)
            next_state = tf.expand_dims(next_state,axis = 0)
            action = tf.expand_dims(action,axis = 0)
            reward = tf.expand_dims(reward,axis = 0)
            done = (done, )
            _ = (_,)


        with tf.GradientTape() as tape:
            target = tf.zeros_like(action,dtype=tf.float32)
            prediction = self.structure(state,training = False)
            # if _:
            #     target = tf.identity(prediction)
            # else:
            #     target = tf.identity(action)
            for idx in range(len(done)):
                Q_new = reward[idx]
                # if not done[idx]
                if  done[idx]:
                    next_state_temp = tf.identity(next_state)
                    next_state_temp = next_state_temp.numpy()
                    next_state_temp = tf.convert_to_tensor(next_state_temp[idx])
                    next_state_temp = tf.expand_dims(next_state_temp,axis = 0)
                    Q_new += self.gamma * np.amax(self.structure(next_state_temp, training = False)) ##chain
                    # temp = target[idx]
                    # temp[np.argmax(action[idx])]
                target = target.numpy()
                if _:
                    target[idx] = tf.identity(prediction[idx]).numpy()
                    target[idx, np.argmax(prediction[idx])] = Q_new
                else:
                    target[idx] = tf.identity(action[idx]).numpy()
                    target[idx, np.argmax(action[idx])] = Q_new

                target[idx] = self.softmax(target[idx])
                target = tf.convert_to_tensor(target)

            loss_value = self.loss(target,prediction)

        grads = tape.gradient(loss_value, self.structure.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.structure.trainable_variables))

        return loss_value,prediction,target

# train = trainer(test,0.001,0.9)
# vax = [ 0. ,0.,0.,-0.07142857 ,-0.21428571 , 0.,1.   ,       0.       ,   0.        ]
# vax = np.array(vax)
# pre = np.array([ 0.          ,0.          ,0.         , 0.35714286 ,-0.14285714  ,1., 0.         , 0.        ,  0.        ])
# print(train.train(vax,[0,0,1],0,pre,0))


