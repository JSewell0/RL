import numpy as np
from numpy.polynomial import Polynomial as P
from tiles3 import tiles, IHT
import pickle
import matplotlib.pyplot as plt
import time 



class TOsarsa:
    def __init__(self,size = 32768,trace_decay=0.9,epsilon=0.4,numruns=5e5):

        self.maxsize = size #max size of index hash table for tile coding
        self.iht = IHT(self.maxsize) #index hash table for tile coding
        self.numruns = numruns #number of runs for training 
        self.term_param = 50 #parameter used for termination condition
        
        self.alpha = 0.1 #step size parameter
        self.Lambda = trace_decay #trace decay parameter for eligibility trace
        self.epsilon = epsilon #probability parameter for epsilon greedy policy
        self.gamma = 0.9 #discount factor 
        
        self.weights = np.zeros(self.maxsize) #weight vector for function approximation
        self.action_space = [-1000,-500,-100,-25,-5,-1,1,5,25,100,500,1000] #set of available actions

        self.reward_arr = [] #stores total reward per episode for analysis
        self.zPos = []

        #function parameters
        self.mu = 3000 
        self.sigma = 300
        self.scale = 1800000

        self.debug = False

    #function used to define curves
    def f(self,x):
        y = np.multiply(np.exp(np.divide(-1*np.power((x-self.mu),2),2*np.power(self.sigma,2))),np.divide(self.scale,2.50663*self.sigma))
        return y


    def get_slope(self,state,action):

        slope = (self.f(self.zPos[-1])-self.f(self.zPos[-2]))/action

        if abs(slope) > 100:
            return 100*slope/abs(slope)
        else:
            return slope


    def get_curve(self,state,action):

        x = np.array([self.zPos[-2],self.zPos[-1]])
        c = P.fit(x,self.f(x),3)

        curve = c.deriv(2)(x[-1])

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
        

    #used to represent a given state-action pair as a binary feature vector using tile coding
    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2000
        yScale = 10/3500
        sScale = 10/200
        cScale = 10/600
        for index in tiles(self.iht,16,[state[0]*yScale, state[1]*sScale, state[2]*cScale,act*aScale]):
            features[index] = 1
        return features
        

    #value function, gives internal value of a given state action pair
    def q(self,state,action):
        features = self.feature_fn(state,action)
        return np.transpose(self.weights)@features

    #takes in state and chooses action based on epsilon greedy policy
    def policy(self,state):

        rng = np.random.default_rng()
        choice = rng.random()

        if choice >= self.epsilon:
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
        
        else:
            act_arr = []
            for action in self.action_space:
                if (action+self.zPos[-1]>0) and (action+self.zPos[-1]<6500):                
                    act_arr.append(action)

            act = rng.choice(act_arr)

            if self.debug:
                print("policy[choice<epsilon]:")
                print(f"-->choice: {choice}")
                print(f"-->epsilon: {self.epsilon}")
                print(f"-->action chosen: {act}\n")
                
            return act
        

    #performs action on given state and outputs new state
    def step(self,state,action):

        new_z = self.zPos[-1] + action
        self.zPos.append(new_z)
        
        new_slope = self.get_slope(state,action)
        new_curve = self.get_curve(state,action)
        reward = -1

        if self.debug:
            print("\nstep function:")
            print(f"-->(fq,action): ({state[0]},{action})")
            print(f"-->new z: {new_z}\n")

        terminated = False
        # if (abs(new_slope)<=1) and (abs(new_z-self.mu)<=self.term_param): 
        if (abs(new_slope)<=1) and (abs(new_z-self.mu)<=self.term_param) and (new_curve>=250):
        # if (abs(new_slope)<=1) and (self.f(new_z)>1700) and  (new_curve>=200):        
            terminated = True
            reward = 10

        return [[self.f(new_z),new_slope,new_curve],reward,terminated]

    
    #initializes a function and lets agent act on said function until termination
    def episode(self,i):
        if (i)%10000:
            print(f"***************************************************************************\n")
            print(f"Starting episode {i}\n")

        param_step = i/self.numruns
        self.alpha = 0.1*(1-param_step)+0.01*(param_step)
        # self.alpha = 0.1
        self.epsilon = 0.15*(1-param_step)+0.05*(param_step)
        # self.term_param = 25*(1-param_step)+10*(param_step)
        
        reward_sum = 0
        t = 0
        
        rng = np.random.default_rng()
        sigma = rng.normal(300,100)
        self.sigma = sigma if sigma>50 else 50
        mu = rng.normal(3200,300)
        self.mu = mu if (mu>0 and mu<6500) else rng.integers(1000,5000)
        scale = rng.normal(1_800_000,500)
        self.scale = scale if self.f(mu)<5000 else 1000000

        self.zPos = []
        initial_z = rng.normal(self.mu,500)
        intial_z = initial_z if (initial_z>0 and initial_z<6500) else 2500
        
        state = [self.f(initial_z),100,1]
        self.zPos.append(initial_z)
        
        action = self.policy(state)

        if self.debug:
            print(f"initial state: {state}")
            print(f"initial action: {action}\n")
        
        feature_vec = self.feature_fn(state,action)
        trace = np.zeros(len(feature_vec))
        Qold = 0
        terminated = False

        while not terminated:
            
            if self.debug:
                print(f"---------------time step {t}---------------")                
            
            new_state, reward, terminated = self.step(state,action)
            new_action = self.policy(new_state)
            reward_sum += reward

            if t == 500:
                terminated = True                        

            if self.debug:
                print(f"state: {new_state}")
                print(f"action: {new_action}\n")

            new_feature_vec = self.feature_fn(new_state,new_action) if (not terminated) else np.zeros(self.maxsize)
            Q = self.q(state,action)
            new_Q = self.q(new_state,new_action)

            TDerr = reward + self.gamma * new_Q - Q
            trace = self.gamma*self.Lambda*trace + (1-self.alpha*self.gamma*self.Lambda*(np.transpose(trace)@feature_vec))*feature_vec
            self.weights = self.weights + self.alpha*(TDerr+Q-Qold)*trace - self.alpha*(Q-Qold)*feature_vec
            Qold = new_Q
            feature_vec = new_feature_vec
            
            if terminated:
                print(f"final action: {action}")
                
            action = new_action
            state = new_state

            if not terminated:
                t += 1
                
            
        self.reward_arr.append(reward_sum)

        if i%10000:
            print(f"\ntime steps: {t}\n")
            print(f"function mu: {self.mu}")
            print(f"final z: {self.zPos[-1]}\n")
            print(f"initial z: {initial_z}")            
            print(f"final state: {new_state}")
            print(f"final reward: {reward_sum}\n")
            print("***************************************************************************\n")

            
def main():
    
    st = time.time()
    
    m = TOsarsa(numruns=5e5)
    # m.debug = True
    for i in range(int(m.numruns)):
        m.episode(i)

    et = time.time()
    
    m.save("v6.9_5e5")
    
    #plots average reward
    i = 0
    avg_reward = np.array([])
    stepnum = int(m.numruns/100)
    while i < m.numruns:
        avg_reward = np.append(avg_reward,np.mean(m.reward_arr[i:i+stepnum]))
        i += stepnum
    
    fig, ax = plt.subplots()

    ax.set_xlabel("episode")
    ax.set_ylabel("avg reward")
    ax.set_title(f"TOsarsa execution time:[{et-st:.1f}]")

    ax.plot(np.linspace(0,m.numruns,100),avg_reward,"-",marker='s',color="royalblue")
    plt.show()
    plt.close()

    




if __name__ == "__main__":
    main()
        




