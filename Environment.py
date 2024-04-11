from typing import Any
import pygame
import random
import time
import numpy as np
import math
snake_body_image = pygame.image.load('decagon_WS.jpg')
snake_head = pygame.image.load('bluegraph.jpg')
background = pygame.image.load('background.png')
FPS=40
WIDTH=560  
HEIGHT=560
pygame.init() 

class game():
    def __init__(self):
        self.rectx = 200
        self.recty = 200
        self.speedx = 40
        self.direction = pygame.K_RIGHT
        self.snake_body = np.array([[self.rectx,self.recty], 
                           [self.rectx-40,self.recty],
                           [self.rectx-80,self.recty],
                           [self.rectx-120,self.recty]])
        self.running = True
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("snake")
        self.inital_pos = np.copy(self.snake_body)[0]
        self.score = 0
        self.move = 0
        self.clock =pygame.time.Clock() 
        self.reset()

    def reset(self):
        self.screen.blit(background,(0,0))
        pygame.display.update()
        self.direction = pygame.K_RIGHT
        self.rectx = 200
        self.recty = 200
        self.snake_body = np.array([[self.rectx,self.recty], 
                           [self.rectx-40,self.recty],
                           [self.rectx-80,self.recty],
                           [self.rectx-120,self.recty]])
        self._fruit_place()
        self.running = True
        self.score = 0
        self.move = 0
        self.draw()
        return self.get_state()
    
    def update(self,actions, *args: Any, **kwargs: Any) -> None:
        self.move +=1
        reward = self.action(actions)
        if self.rectx == self.fruit_pos[0] and self.recty == self.fruit_pos[1]:
            self.snake_body = np.insert(self.snake_body,0,values=[self.rectx,self.recty],axis=0)
            reward = 100
            self.score +=10
            self._fruit_place()
            self.draw()
            self.move = 0
            self.inital_pos = np.copy(self.snake_body)[0]
            return self.get_state(),reward,self.running
        
        if [self.rectx,self.recty] in self.snake_body.tolist()[:-1] or self.rectx>520 or self.rectx<0 or self.recty>520 or self.recty<0:
            self.running = False
            reward = -30
            return self.get_state(),reward,self.running
        
        elif self.move > 100:
            self.running = False
            reward = -20
            return self.get_state(),reward,self.running

        self.snake_body = np.insert(self.snake_body,0,values=[self.rectx,self.recty],axis=0)
        self.snake_body = np.delete(self.snake_body,np.size(self.snake_body,0)-1,axis = 0)
        self.draw()
        return self.get_state(),reward,self.running
    
    def current_scores(self):
        return self.score
    
    def draw(self):
        self.clock.tick(FPS)
        self.screen.blit(background,(0,0))
        pygame.display.update()
        pygame.draw.rect(self.screen, (255,0,0), (self.fruit_pos[0],self.fruit_pos[1],40,40))
        self.screen.blit(snake_head,(self.snake_body[0][0]+1,self.snake_body[0][1]+1))
        for pos in self.snake_body[1:]:
            self.screen.blit(snake_body_image,(pos[0]+1,pos[1]+1))
        pygame.display.update()

    def _fruit_place(self):
        posX = round((WIDTH/40-1)*random.random())*40
        posY = round((HEIGHT/40-1)*random.random())*40
        while [posX,posY] in self.snake_body:
            posX = round((WIDTH/40-1)*random.random())*40
            posY = round((HEIGHT/40-1)*random.random())*40
        self.fruit_pos = np.array([posX,posY])   

    def action(self,actions):
        action_lists = [pygame.K_UP,pygame.K_RIGHT,pygame.K_DOWN,pygame.K_LEFT]
        reward = 0
        if actions == [1,0,0]:
            key_pressed = self.direction
        elif actions == [0,1,0]:
            index = (action_lists.index(self.direction)+1)%4
            key_pressed = action_lists[index]
        else:
            index = (action_lists.index(self.direction)-1)%4
            key_pressed = action_lists[index]
        if key_pressed==pygame.K_RIGHT:
            temp = self.rectx
            self.rectx = self.rectx+ self.speedx
            if abs(self.rectx-self.fruit_pos[0]) < abs(temp-self.fruit_pos[0]):
                reward = 1
            else:
                reward = -1
        elif key_pressed==pygame.K_LEFT:
            temp = self.rectx
            self.rectx = self.rectx - self.speedx
            if abs(self.rectx-self.fruit_pos[0]) < abs(temp-self.fruit_pos[0]):
                reward = 1
            else:
                reward = -1
        elif key_pressed==pygame.K_UP:
            temp = self.recty
            self.recty = self.recty - self.speedx
            if abs(self.recty-self.fruit_pos[1]) < abs(temp-self.fruit_pos[1]):
                reward = 1
            else:
                reward = -1
        elif key_pressed==pygame.K_DOWN:
            temp = self.recty
            self.recty = self.recty + self.speedx
            if abs(self.recty-self.fruit_pos[1]) < abs(temp-self.fruit_pos[1]):
                reward = 1
            else:
                reward = -1
        self.direction = key_pressed
        return reward

    def get_state(self):
        snake_pos = np.array([self.rectx, self.recty])
        difference_pos = abs(snake_pos-self.fruit_pos)/40
        near_have_fruit = [
            self.recty - self.fruit_pos[1] == 40, ## fruit on uper
            self.rectx - self.fruit_pos[0] == -40, ## fruit on right 
            self.rectx - self.fruit_pos[0] == 40 , ## fruit on left
            self.recty - self.fruit_pos[1] == -40
        ]
        dict = {pygame.K_UP:[1,0,0,0],pygame.K_RIGHT:[0,1,0,0],pygame.K_DOWN:[0,0,1,0],pygame.K_LEFT:[0,0,0,1]}

        current_direction_TF = dict[self.direction]
        direction = np.array(current_direction_TF)
        danger1 = [
            (current_direction_TF[0] and self.recty == 0) or
            (current_direction_TF[1] and self.rectx == 520) or
            (current_direction_TF[2] and self.recty == 520) or 
            (current_direction_TF[3] and self.rectx == 0),

            (current_direction_TF[0] and self.rectx == 520) or
            (current_direction_TF[1] and self.recty == 520) or
            (current_direction_TF[2] and self.rectx == 0) or 
            (current_direction_TF[3] and self.recty == 0),

            (current_direction_TF[0] and self.rectx == 0) or
            (current_direction_TF[1] and self.recty == 0) or
            (current_direction_TF[2] and self.rectx == 520) or 
            (current_direction_TF[3] and self.recty == 520),
            
        ]

        danger2 = [
            (current_direction_TF[0] and [self.rectx,self.recty-20] in self.snake_body) or ##upper_have body
            (current_direction_TF[1] and [self.rectx+20,self.recty] in self.snake_body) or
            (current_direction_TF[2] and [self.rectx,self.recty+20] in self.snake_body) or
            (current_direction_TF[3] and [self.rectx-20,self.recty] in self.snake_body),

            (current_direction_TF[0] and [self.rectx+20,self.recty] in self.snake_body) or ##upper_have body
            (current_direction_TF[1] and [self.rectx,self.recty+20] in self.snake_body) or
            (current_direction_TF[2] and [self.rectx-20,self.recty] in self.snake_body) or
            (current_direction_TF[3] and [self.rectx,self.recty-20] in self.snake_body),

            (current_direction_TF[0] and [self.rectx-20,self.recty] in self.snake_body) or ##upper_have body
            (current_direction_TF[1] and [self.rectx,self.recty-20] in self.snake_body) or
            (current_direction_TF[2] and [self.rectx+20,self.recty] in self.snake_body) or
            (current_direction_TF[3] and [self.rectx,self.recty+20] in self.snake_body),
        ]
        fruit_location = [self.fruit_pos[0] < self.rectx ,
                          self.fruit_pos[0] > self.rectx, 
                          self.fruit_pos[1] > self.recty, 
                          self.fruit_pos[1] < self.recty]
        
        difference_pos = np.insert(difference_pos,2,near_have_fruit,axis = 0)
        difference_pos = np.insert(difference_pos,6,fruit_location,axis = 0)
        group = np.insert(difference_pos,10,direction,axis = 0)
        group = np.insert(group,14,danger1,axis=0)
        return np.insert(group,17,danger2,axis=0)
        # state = [[0] * 14 for _ in range(14)]
        # for x,y in self.snake_body.tolist():
        #     state[x//40][y//40] = 1
        # state[self.fruit_pos[0]//40][self.fruit_pos[1]//40] = -1
        # state = np.array(state)
        # return state.reshape((14,14,1))
    
    def quit(self):
        pygame.quit()
        quit()

# Game = game()
# print(Game.get_state())
# x, _ ,_  = Game.update([0,0,1])
# print(x)