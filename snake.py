from typing import Any
import pygame
import random
import time
snake_body_image = pygame.image.load('decagon_WS.jpg')
snake_head = pygame.image.load('bluegraph.jpg')
background = pygame.image.load('background.png')
FPS=5
WIDTH=560  
HEIGHT=560
pygame.init() 
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("snake")
clock=pygame.time.Clock() 
impossible_direction = [[pygame.K_DOWN,pygame.K_UP],[pygame.K_UP,pygame.K_DOWN],[pygame.K_LEFT,pygame.K_RIGHT],[pygame.K_RIGHT,pygame.K_LEFT]]
fruit_pos = [400,400]
random.seed(1000)


def game_over():

	my_font = pygame.font.SysFont('times new roman', 50)
	
	game_over_surface = my_font.render(
		'Dead!', True, (255,0,0))
	
	game_over_rect = game_over_surface.get_rect()
	game_over_rect.midtop = (WIDTH/2, HEIGHT/4)
	screen.blit(game_over_surface, game_over_rect)
	pygame.display.flip()
	
	# after 2 seconds we will quit the program
	time.sleep(2)
	
	# deactivating pygame library
	pygame.quit()
	
	# quit the program
	quit()


class snake(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40,40))
        self.image.fill((0,255,0))
        self.rect =self.image.get_rect() # for circulating the photo (self.image)
        # self.snack = [[self.rect.x, self.rect.y], [self.rect.x-40,self.rect.y],[self.rect.x-80,self.rect.y],[self.rect.x-120,self.rect.y]]
        self.rect.x = 200
        self.rect.y = 200
        self.speedx = 40
        self.direction = pygame.K_RIGHT
        self.snake_body = [[self.rect.x,self.rect.y], 
                           [self.rect.x-40,self.rect.y],
                           [self.rect.x-80,self.rect.y],
                           [self.rect.x-120,self.rect.y]]
    def update(self,key_pressed, *args: Any, **kwargs: Any) -> None:
        global fruit_pos
        if key_pressed==pygame.K_RIGHT:
            self.rect.x += self.speedx
        elif key_pressed==pygame.K_LEFT:
            self.rect.x -= self.speedx
        elif key_pressed==pygame.K_UP:
            self.rect.y -= self.speedx
        elif key_pressed==pygame.K_DOWN:
            self.rect.y += self.speedx
        self.direction = key_pressed
        if self.rect.x == fruit_pos[0] and self.rect.y == fruit_pos[1]:
            self.snake_body.insert(0,[self.rect.x,self.rect.y])
            posX = round((WIDTH/40-1)*random.random())*40
            posY = round((HEIGHT/40-1)*random.random())*40
            while [posX,posY] in self.snake_body:
                posX = round((WIDTH/40-1)*random.random())*40
                posY = round((HEIGHT/40-1)*random.random())*40
            fruit_pos = [posX,posY]
            return
        if [self.rect.x,self.rect.y] in self.snake_body[:-1] or self.rect.x>520 or self.rect.x<0 or self.rect.y>520 or self.rect.y<0:
            game_over()
        self.snake_body.insert(0,[self.rect.x,self.rect.y])
        self.snake_body.pop()
        return super().update(*args, **kwargs)
    def draw(self):
        pygame.draw.rect(screen, (255,0,0), (fruit_pos[0],fruit_pos[1],40,40))
        screen.blit(snake_head,(self.snake_body[0][0]+1,self.snake_body[0][1]+1))
        for pos in self.snake_body[1:]:
            screen.blit(snake_body_image,(pos[0]+1,pos[1]+1))

# whole_snack = pygame.sprite.Group()
# Snack = snack()
# whole_snack.add(Snack)
Snake = snake()
running_time = True


screen.blit(background,(0,0))
pygame.display.update()
print(Snake.rect.x,Snake.rect.y)
while running_time:
    direction = 0
    clock.tick(FPS)
    screen.blit(background,(0,0))
    for event in pygame.event.get():
        if event.type ==pygame.QUIT:
            running_time = False
        elif event.type == pygame.KEYDOWN:
            direction = event.key
    if direction == 0 or [direction,Snake.direction] in impossible_direction:
        Snake.update(Snake.direction)
    else:
        Snake.update(direction)        
    Snake.draw()
    pygame.display.update()

