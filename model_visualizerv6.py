import numpy as np
from numpy.polynomial import Polynomial as P
from tiles3 import tiles, IHT
import pickle
import matplotlib.pyplot as plt
import time 
import matplotlib.animation as animation


class TOsarsa:
    def __init__(self,size=32768,trace_decay=0.9,epsilon=0.4,model=""):

        self.maxsize = size
        
        if len(model) == 0:
            self.iht = IHT(self.maxsize)
            self.weights = np.zeros(self.maxsize)            
        else:
            with open(model,'rb') as f:
                obj = pickle.load(f)
                self.weights = obj[0]
                self.iht = obj[1]
                
            
        self.alpha = 0.1
        self.Lambda = trace_decay
        self.epsilon = epsilon
        self.gamma = 0.9
        self.end_time = 100

        self.action_space = [-1000,-500,-100,-25,-5,5,25,100,500,1000]
        self.fq_arr = np.array([])
        self.zPos = []

        self.point = 0

        self.mu = 3000
        self.sigma = 300
        self.scale = 1800000

        self.debug = False

    def f(self,x):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-self.mu),2),2*np.power(self.sigma,2))),np.divide(self.scale,2.50663*self.sigma))
        return y


    def get_slope(self,state,action):

        slope = (self.f(self.zPos[-1]+action)-self.f(self.zPos[-2]))/action

        if abs(slope) > 100:
            return 100*slope/abs(slope)
        else:
            return slope


    def get_curve(self,state,action):

        x = np.array([self.zPos[-2],self.zPos[-1]])
        c = P.fit(x,self.f(x),3)

        curve = c.deriv(2)(x[-1])
        # if not term_check:
            # print(f"\ntime step: {t}\nx_arr: {x}\ncurve: {curve}")

        if curve >= 0 and curve <= 600:
            return curve
        
        elif curve < 0:
            return 0
        
        elif curve > 600:
            return 600

    def update(self,frame):

        z = self.zPos[frame-1:frame]
        fq = self.fq_arr[frame-1:frame]

        self.point.set_xdata(z)
        self.point.set_ydata(fq)

    
    def save(self,version="v1"):
        
        with open(f"model_{version}.pickle","wb") as f:
            pickle.dump([self.weights,self.iht],f)                

        print("Saved!")
    

    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2000
        yScale = 10/6500
        sScale = 10/200
        cScale = 10/600
        # cScale = 10
        for index in tiles(self.iht,16,[state[0]*yScale, state[1]*sScale, state[2]*cScale,act*aScale]):
            features[index] = 1
        return features
        

    def q(self,state,action):
        features = self.feature_fn(state,action)
        return np.transpose(self.weights)@features

    
    def policy(self,state):

        val_arr = np.array([])

        for action in self.action_space:
            if (action+self.zPos[-1]>0) and (action+self.zPos[-1]<6500):
                val_arr = np.append(val_arr, self.q(state,action))
            else:
                val_arr = np.append(val_arr, -np.inf)

        act = self.action_space[np.argmax(val_arr)]

        if self.debug:
            print("policy[choice>epsilon]:")
            print(f"-->choice: {choice}")
            print(f"-->epsilon: {self.epsilon}")                
            print(f"-->val arr: {val_arr}\n")
            print(f"-->action chosen: {act}")

        return act
        

    def step(self,state,action):

        new_z = self.zPos[-1] + action
        self.zPos.append(new_z)
        new_slope = self.get_slope(state,action)
        new_curve = self.get_curve(state,action)
        # new_curve = 1
        reward = -1

        if self.debug:
            print("\nstep function:")
            print(f"-->(z,action): ({self.zPos[-1]},{action})")
            print(f"-->new z: {new_z}")
            print(f"-->slope: {new_slope}")
            print(f"-->curve: {new_curve}\n")

        terminated = False
        if (abs(new_slope)<=1) and (self.f(new_z)>1500) and (abs(action)<250):
            terminated = True
            reward = 10
            
        # terminated = False
        # if (abs(new_slope)<=1) and (new_curve>=350):
        #     terminated = True
        #     reward = 10

        return [[self.f(new_z),new_slope,new_curve],reward,terminated]


    def episode(self,i):

        fig, ax = plt.subplots()
        
        reward_sum = 0
        t = 0
        self.zPos = []
        self.fq_arr = np.array([])
        
        rng = np.random.default_rng()
        sigma = rng.normal(300,100)
        self.sigma = sigma if sigma>1 else 1
        mu = rng.normal(3200,250)
        self.mu = mu if (mu>0 and mu<6500) else rng.integers(1000,5000)
        scale = rng.normal(1_800_000,500)
        self.scale = scale if self.f(mu)<5000 else 1000000

        initial_z = rng.normal(self.mu,250)
        self.zPos.append(initial_z)
        
        state = [self.f(initial_z),100,1]
        
        action = self.policy(state)

        if self.debug:
            print(f"initial state: {state}")
            print(f"initial action: {action}\n")

        terminated = False
        converged = True
        
        self.fq_arr = np.append(self.fq_arr,self.f(initial_z))

        z_space = np.arange(0,6501)
        line = plt.plot(z_space,self.f(z_space),c='k',label='function')
        self.point = ax.plot(initial_z,self.f(initial_z),marker="X",mec="r",mfc="r",lw=0,label='agent position')[0]
        ax.set(xlim=[0, 6500], ylim=[-100, max(self.f(z_space))+200], xlabel='z pos', ylabel='fq val')
        ax.legend()
        
        while not terminated:
            
            if self.debug:
                print(f"---------------time step {t}---------------")                
            
            new_state, reward, terminated = self.step(state,action)
            new_action = self.policy(new_state)
            reward_sum += reward

            self.fq_arr = np.append(self.fq_arr,new_state[0])           

            if t == self.end_time:
                terminated = True
                converged = False

            if self.debug:
                print(f"state: {new_state}")
                print(f"action: {new_action}\n")
                
            state = new_state
            action = new_action

                
            if not terminated:
                t += 1

        print(f"\nnumber of time steps: {len(self.zPos)-2}\n")

        ani = animation.FuncAnimation(fig=fig, func=self.update, frames=len(self.zPos), interval=250, repeat=True)
        plt.show()
        


                
def main():
    

    m = TOsarsa(model='model_v5.4.all/model_v5,4,3_1e6.pickle')
    m.debug = False
    m.end_time = 500
    numruns = 10

    for i in range(int(numruns)):
        m.episode(i)



if __name__ == "__main__":
    main()
        




