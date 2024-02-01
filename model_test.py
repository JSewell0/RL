import numpy as np
from numpy.polynomial import Polynomial as P
from tiles3 import tiles, IHT
import pickle
import matplotlib.pyplot as plt
import time 



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

        self.action_space = [-1000,-500,-100,-25,-5,5,25,100,500,1000]
        self.reward_arr = []
        self.time_arr = []
        self.diff_arr = []

        self.mu = 3000
        self.sigma = 300
        self.scale = 1800000

        self.debug = False

    def f(self,x):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-self.mu),2),2*np.power(self.sigma,2))),np.divide(self.scale,2.50663*self.sigma))
        return y


    def get_slope(self,state,action):

        slope = (self.f(state[0]+action)-self.f(state[0]))/action

        if abs(slope) > 100:
            return 100*slope/abs(slope)
        else:
            return slope


    def get_curve(self,state,action):

        x = np.array([state[0],state[0]+action])
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

    
    def save(self,version="v1"):
        
        with open(f"model_{version}.pickle","wb") as f:
            pickle.dump([self.weights,self.iht],f)                

        print("Saved!")

    def plot_reward(self,numruns,timediff):
        i = 0
        avg_reward = np.array([])
        stepnum = int(numruns/50)
        while i < numruns:
            avg_reward = np.append(avg_reward,np.mean(self.reward_arr[i:i+stepnum]))
            i += stepnum

        fig, ax = plt.subplots()

        ax.set_xlabel("episode")
        ax.set_ylabel("avg reward")
        ax.set_title(f"TOsarsa execution time:[{timediff:.1f}]")

        ax.plot(np.linspace(0,numruns,50),avg_reward,"-",marker='s',color="royalblue")
        plt.show()
        plt.close()

    def plot_time(self,plot_wins = False):
        wSum = 0
        lSum = 0
        for time in self.time_arr:
            if time>199:
                lSum += 1
            else:
                wSum += 1
                
        print(f"{wSum/len(self.time_arr)*100}% converged")
        print(f"{lSum/len(self.time_arr)*100}% failed")

        fig, ax = plt.subplots()

        ax.set_xlabel("time steps")
        ax.set_ylabel("count")
        ax.set_title(f"time steps for completion")        

        if plot_wins:
            ax.hist(self.time_arr,range = (min(self.time_arr),200),bins=25,log=True)#,density=True,log=True)
        else:
            ax.hist(self.time_arr,bins=25,density=True,log=True)
        plt.show()
        plt.close()


    def plot_diff(self):

        fig, ax = plt.subplots()

        ax.set_xlabel("distance")
        ax.set_ylabel("count")
        ax.set_title(f"final state-Mu difference")        

        ax.hist(self.diff_arr,bins=25,log=True)#,density=True,log=True)
        plt.show()
        plt.close()        
        

    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2000
        pScale = 10/6500
        sScale = 10/200
        cScale = 10/600
        for index in tiles(self.iht,16,[state[0]*pScale, state[1]*sScale, state[2]*cScale,act*aScale]):
            features[index] = 1
        return features
        

    def q(self,state,action):
        features = self.feature_fn(state,action)
        return np.transpose(self.weights)@features

    
    def policy(self,state):

        choice = 1

        if choice >= self.epsilon:
            val_arr = np.array([])
            
            for action in self.action_space:
                if (action+state[0]>0) and (action+state[0]<6500):
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

        new_z = state[0] + action
        new_slope = self.get_slope(state,action)
        new_curve = self.get_curve(state,action)
        reward = -1

        if self.debug:
            print("\nstep function:")
            print(f"-->(z,action): ({state[0]},{action})")
            print(f"-->new z: {new_z}\n")

        terminated = False
        if (abs(new_slope)<=1) and (new_curve>=300):
            terminated = True
            reward = 10

        return [[new_z,new_slope,new_curve],reward,terminated]


    def episode(self,i):
        
        reward_sum = 0
        t = 0
        
        rng = np.random.default_rng()
        sigma = rng.normal(300,100)
        self.sigma = sigma if sigma>1 else 1
        mu = rng.normal(3200,250)
        self.mu = mu if (mu>0 and mu<6500) else rng.integers(1000,5000)
        scale = rng.normal(1_800_000,500)
        self.scale = scale if self.f(mu)<5000 else 1000000

        initial_z = rng.normal(self.mu,250)
        state = [initial_z,100,0]
        
        action = self.policy(state)

        if self.debug:
            print(f"initial state: {state}")
            print(f"initial action: {action}\n")

        terminated = False
        converged = True

        while not terminated:
            
            if self.debug:
                print(f"---------------time step {t}---------------")                
            
            new_state, reward, terminated = self.step(state,action)
            new_action = self.policy(new_state)
            reward_sum += reward

            if t == 200:
                terminated = True
                converged = False

            if self.debug:
                print(f"state: {new_state}")
                print(f"action: {new_action}\n")
                
            state = new_state
            action = new_action

                
            if not terminated:
                t += 1

        self.time_arr.append(t)
        if converged:
            self.diff_arr.append(abs(state[0]-self.mu))
        self.reward_arr.append(reward_sum)

            
def main():
    

    m = TOsarsa(model='model_v5_2e5.pickle')
    m.debug = False
    numruns = 5000
    st = time.time()
    for i in range(int(numruns)):
        m.episode(i)

    et = time.time()

    m.plot_time(plot_wins=True)
    m.plot_reward(numruns,et-st)
    m.plot_diff()


if __name__ == "__main__":
    main()
        




