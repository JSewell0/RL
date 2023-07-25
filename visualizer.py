import numpy as np
import matplotlib.pyplot as plt
import pickle
import pygame

class racecar:
    def __init__(self,position=[0,0],velocity=[0,0]):
        self.position = position
        self.velocity = velocity

    def accelerate(self,deltav,flag):
        if not flag:
            self.velocity[0] += deltav[0]
            self.velocity[1] += deltav[1]

    def move(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def return_to_start(self):
        starts = [[i,1] for i in range(4,10,1)]
        rng = np.random.default_rng()
        start_pos = rng.integers(0,len(starts))
        
        self.position[0] = starts[start_pos][0]
        self.position[1] = starts[start_pos][1]
        
        self.velocity[0] = 0
        self.velocity[1] = 0



def make_track():    
    ###Defining track###
    x, y = np.meshgrid(range(100),range(100),indexing="ij")
    track = {}
    nx = np.full((100,100),0)
    ny = np.full((100,100),0)

    for i in range(100):
        for j in range(100):
            ### space around track ###        
            if i == 0 or i >= 18:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if j >= 33 or j == 0:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            ### bottom right space ###
            if i > 9 and j < 26:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i > 10 and j == 26:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            ### bottom left space ###
            if i <= 3 and j <= 3:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 2 and j <= 10:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 1 and j <= 18:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            ### top right space ###
            if i <= 3 and j >= 32:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 2 and j >= 30:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 1 and j >= 29:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]

    gridx = x - nx
    gridy = y - ny

    for i in range(100):
        for j in range(100):
            if gridx[i][j] != 0 and gridy[i][j] != 0:
                if i  == 17  and j < 33 and j >= 27:
                    track[(i,j)] = 2
                elif i < 10  and i >= 4 and j == 1:
                    track[(i,j)] = 1
                else:
                    track[(i,j)] = 0
            else:
                track[(i,j)] = -1
                

    return track


class Visualizer:
    
    #HELPFUL FUNCTIONS
    
    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack")
    
    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 9
        self.width = 100*self.cell_edge
        self.height = 100*self.cell_edge
        self.create_window()
        self.window = True

    def close_window(self):
        self.window = False
        pygame.quit()

    def draw(self, state = np.array([])):
        self.display.fill(0)
        for i in range(100):
            for j in range(100):
                if self.data[i,j]!=-1:
                    if self.data[i,j] == 0:
                        color = (255,0,0)
                    elif self.data[i,j] == 1:
                        color = (255,255,0)
                    elif self.data[i,j] == 2:
                        color = (0,255,0)
                    pygame.draw.rect(self.display,color,((j*self.cell_edge,i*self.cell_edge),(self.cell_edge,self.cell_edge)),1)
        
        if len(state)>0:
            pygame.draw.rect(self.display,(0,0,255),((state[1]*self.cell_edge,state[0]*self.cell_edge),(self.cell_edge,self.cell_edge)),1)
        
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and pygame.K_0:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.loop = False
                
        return None
        
    def visualize_racetrack(self, state = np.array([])):
        '''
        Draws Racetrack in a pygame window
        '''
        if self.window == False:
            self.setup()
        self.loop = True
        while(self.loop):
            ret = self.draw(state)
            if ret!=None:
                return ret
    
    #CONSTRUCTOR
    def __init__(self,data):
        self.data = data
        self.window = False



with open("q_pi_racecar5e4v2.pickle","rb") as f:
    value_fn, pi = pickle.load(f)
track = make_track()

w = Visualizer(track)
car = racecar()
car.return_to_start()
running = True
car.position = [12,30]
car.velocity = [2, 0]


while running:
    [x, y] = car.position
    [vx, vy] = car.velocity


    # print(f"initial: {x},{y},{vx},{vy}")
    
    ax, ay = pi[(x,y,vx,vy)]

    # print(f"action: {(ax,ay)}")

    #check for finish
    for xstep in range(x,x+vx+1,1):
        for ystep in range(y,y+vy+1):
            if  track[(xstep,ystep)] == 2:
                w.visualize_racetrack([x+vx,y+vy])                
                running = False

    
    #check if car is still in track
    break_flag = False
    if track[(np.add(x,vx),np.add(y,vy))] == -1:
        break_flag = True
        car.return_to_start()        

    w.visualize_racetrack([x,y])    
    car.move()                
    car.accelerate([ax,ay],break_flag)
    # print(f"end of step: {car.position} {car.velocity}\n")




