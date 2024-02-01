import numpy as np
from tiles3 import tiles, IHT
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import time 



class TOsarsa:
    def __init__(self,size = 16384,trace_decay=0.92,epsilon=0.7):

        self.env = gym.make("MountainCar-v0")#, render_mode="human")        
        self.maxsize = size
        self.iht = IHT(self.maxsize)    
        self.alpha = 1/self.maxsize
        self.Lambda = trace_decay
        self.epsilon = epsilon
        self.weights = np.zeros(self.maxsize)
        self.action_space = [0,1,2]
        self.gamma = 0.9
        self.reward_arr = []


    def save(self,version="v1"):
        
        with open(f"mountaincar_{version}.pickle","wb") as f:
            pickle.dump([self.weights,self.iht],f)                

        print("Saved!")
        

    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2
        pScale = 10/1.8
        vScale = 10/0.14
        for index in tiles(self.iht,16,[state[0]*pScale, state[1]*vScale, act*aScale]):
            features[index] = 1
        return features
        

    def q(self,state,action):
        features = self.feature_fn(state,action)
        return np.transpose(self.weights)@features

    
    def policy(self,state):

        rng = np.random.default_rng()
        choice = rng.random()

        if choice >= self.epsilon:
            val_arr = np.array([])
            for action in self.action_space:
                val_arr = np.append(val_arr, self.q(state,action))
            return np.argmax(val_arr)
        else:
            return self.env.action_space.sample()      
        

    def episode(self,i):
        print(f"***************************************************************************\n")
        print(f"Starting episode {i+1}\n")
        
        reward_sum = 0
        state, info = self.env.reset()

        action = self.policy(state)
        feature_vec = self.feature_fn(state,action)
        trace = np.zeros(len(feature_vec))
        Qold = 0
        terminated = False

        while not terminated:
            
            new_state, reward, terminated, truncated, info = self.env.step(action)
            new_action = self.policy(new_state)
            reward_sum += reward

            new_feature_vec = self.feature_fn(new_state,new_action) if (not terminated) else np.zeros(self.maxsize)
            Q = self.q(state,action)
            new_Q = self.q(new_state,new_action)

            TDerr = reward + self.gamma * new_Q - Q
            trace = self.gamma*self.Lambda*trace + (1-self.alpha*self.gamma*self.Lambda*(np.transpose(trace)@feature_vec))*feature_vec
            self.weights = self.weights + self.alpha*(TDerr+Q-Qold)*trace - self.alpha*(Q-Qold)*feature_vec
            Qold = new_Q
            feature_vec = new_feature_vec
            action = new_action
            
        self.reward_arr.append(reward_sum)
        print("***************************************************************************\n")

            
def main():
    
    st = time.time()
    m = TOsarsa()
    for i in range(1000):
        m.episode(i)

    et = time.time()    

    m.env = gym.make("MountainCar-v0", render_mode="human")
    m.epsilon = 0.01
    m.episode(1110)

    i = 0
    avg_reward = np.array([])
    while i < 1000:
        avg_reward = np.append(avg_reward,np.mean(m.reward_arr[i:i+49]))
        i += 50
    
    fig, ax = plt.subplots()

    ax.set_xlabel("episode")
    ax.set_ylabel("avg reward")
    ax.set_title(f"TOsarsa execution time:[{et-st:.1f}]")

    ax.plot(np.linspace(0,1000,20),avg_reward,"-",marker='s',color="royalblue")
    plt.show()
    plt.close()



if __name__ == "__main__":
    main()
        




