import numpy as np
import matplotlib.pyplot as plt
import pickle


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

def b_policy(x,y,vx,vy,q,pi,epsilon=0.1):
    
    rng = np.random.default_rng()
    action_chance = 0
    
    action_space = q[(x,y,vx,vy)]
    possible_actions = []

    for keys, value in action_space.items():
        p_ax, p_ay = keys
        if vx + p_ax < 5 and vy + p_ay < 5 and vx + p_ax >= 0 and vy + p_ay >= 0 and not(vx+p_ax==0 and vy+p_ay==0):
            possible_actions.append((p_ax,p_ay))

    # print(f"state: {x},{y},{vx},{vy}")
    # print(f"possible_actions: {possible_actions}")
    greedy_action = pi[(x,y,vx,vy)]
    # print(f"greedy_action: {greedy_action}")
    choice = rng.random()
    if choice >= epsilon and greedy_action in possible_actions:
        # print("GREEDY\n")
        action_chance = 1-epsilon+(epsilon/len(possible_actions))
        ax, ay = greedy_action
    else:

        action_chance = epsilon/len(possible_actions)
        ax, ay = rng.choice(possible_actions)
        # print(f"explored action: {(ax,ay)}\n")        
        
    return ax, ay, action_chance

def episode(track,q,pi):

    policy = pi
    value_fn = q
    car = racecar()
    car.return_to_start()
    steps = {}
    counter = 0
    finish_line = [[17,i] for i in range(27,33,1)]
    rng = np.random.default_rng()
    print("starting episode... \n")
    while True:
        x, y = car.position
        vx, vy = car.velocity
        
        # print(f"Initial:{x}, {y}, {vx}, {vy}")
        action_space = value_fn[(x,y,vx,vy)]
        ax, ay, action_chance = b_policy(x,y,vx,vy,value_fn,policy)

        #randomly set velocity increments to zero
        choice = rng.choice([0,1],p=[0.1,0.9])
        ax, ay = (0,0) if choice == 0 else (ax,ay)
        # print(f"action: {(ax,ay)}")
        
        #check for finish
        for finish_cell in finish_line:
            for xstep in range(x,x+vx+1,1):
                for ystep in range(y,y+vy+1,1):
                    # print(f"finish cell: {finish_cell} check cell: {(xstep,ystep)}")
                    if [xstep,ystep] == finish_cell:
                        steps[counter] = [(x,y,vx,vy),(ax,ay),0,action_chance]
                        print(f"finished track on step: {counter}")
                        return steps

        #check if car is still in track
        break_flag = False
        try:
            track[(np.add(x,vx) ,np.add(y,vy))]
        except:
            break_flag = True
            # print("hit wall!")            
            car.return_to_start()
            
        steps[counter] = [(x,y,vx,vy),(ax,ay),-1,action_chance]
        car.move()        
        car.accelerate([ax,ay],break_flag)
        counter += 1
        # print(f"end of step:{car.position}, {car.velocity}\n")
        
def init_pi(track,value_fn):
    pi = {}
    for pos in track.keys():
        x_pos, y_pos = pos
        for vx in range(0,5,1):
            for vy in range(0,5,1):
                v = list(value_fn[(x_pos,y_pos,vx,vy)].values())
                k = list(value_fn[(x_pos,y_pos,vx,vy)].keys())
                pi[(x_pos,y_pos,vx,vy)] = k[v.index(max(v))]
    return pi

    
def make_track():    
    ###Defining track###
    x, y = np.meshgrid(range(19),range(34),indexing="ij")

    nx = np.full((19,34),0)
    ny = np.full((19,34),0)

    for i in range(19):
        for j in range(34):
            ### space around track ###        
            if i == 0 or i == 18:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if j == 33 or j == 0:
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
            if i <= 3 and j <= 2:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 2 and j <= 9:
                nx[i][j] = x[i][j]
                ny[i][j] = y[i][j]
            if i <= 1 and j <= 17:
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
    track = {}
    for i in range(19):
        for j in range(34):
            if gridx[i][j] != 0 and gridy[i][j] != 0:
                track[(i,j)] = True   

    return track

def control(track):
    weight_sum = {}
    value_fn = {}
    discount = 0.9
    reward = np.array([])
    #init weight_sum and value_fn
    for pos in track.keys():
        rng = np.random.default_rng()
        x_pos, y_pos = pos

        for vx in range(0,5,1):
            for vy in range(0,5,1):
                temp1 = {}
                temp2 = {}
                for delta_vx in range(-1,2,1):
                    for delta_vy in range(-1,2,1):
                        if vx + delta_vx < 5 and vy + delta_vy < 5 and vx + delta_vx >= 0 and vy + delta_vy >= 0 and not(vx+delta_vx==0 and vy+delta_vy==0):                        
                            temp1[(delta_vx,delta_vy)] = 0
                            temp2[(delta_vx,delta_vy)] = rng.integers(0,10)                        
                weight_sum[(x_pos,y_pos,vx,vy)] = temp1
                value_fn[(x_pos,y_pos,vx,vy)] = temp2
    #Start eval_loop
    pi = init_pi(track,value_fn)
    for i in range(50_000):
        print("************************************************************")
        print(f"starting big loop: {i+1}")
        returns = 0
        weight = 1
        steps = episode(track,value_fn,pi)
        reward = np.append(reward,-1*len(steps))
        for j, items in enumerate(list(reversed(steps.items()))):
            key, vals = items
            returns = discount*returns + vals[2]
            weight_sum[vals[0]][vals[1]] += weight
            value_fn[vals[0]][vals[1]] += (weight/(weight_sum[vals[0]][vals[1]]))*(returns-value_fn[vals[0]][vals[1]])
            v = list(value_fn[vals[0]].values())
            k = list(value_fn[vals[0]].keys())
            pi[vals[0]] = k[v.index(max(v))]
            if vals[1] != pi[vals[0]]:
                print(f"num of improvement loops: {j+1}")
                print("************************************************************\n")                
                break
            weight = weight/vals[3]
    return value_fn, pi, reward

            
            
rng = np.random.default_rng()
track = make_track()
value_fn, pi, rewards = control(track)

with open("q_pi_racecar5e4v2.pickle","wb") as f:
    pickle.dump((value_fn,pi),f)

ax, fig = plt.subplots(figsize=(30,15))
x = np.arange(1,len(rewards)+1)
plt.plot(x*10, rewards, linewidth=0.5, color = '#BB8FCE')
plt.xlabel('Episode number', size = 20)
plt.ylabel('Reward',size = 20)
plt.title('Plot of Reward vs Episode Number',size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.savefig('RewardGraph5e4v2.png')
plt.close()


