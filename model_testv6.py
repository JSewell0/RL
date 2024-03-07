import numpy as np
from numpy.polynomial import Polynomial as P
from tiles3 import tiles, IHT
import pickle
import matplotlib.pyplot as plt
import time 



class TOsarsa:
    def __init__(self,size=32768,epsilon=0.4,model=""):

        self.maxsize = size
        
        if len(model) == 0:
            self.iht = IHT(self.maxsize)
            self.weights = np.zeros(self.maxsize)            
        else:
            with open(model,'rb') as f:
                obj = pickle.load(f)
                self.weights = obj[0]
                self.iht = obj[1]
                       
        self.epsilon = epsilon
        self.end_time = 100

        self.action_space = [-1000,-500,-100,-25,-5,-1,1,5,25,100,500,1000]
        self.reward_arr = []
        self.time_arr = []
        self.diff_arr = []
        self.final_arr = []
        self.zPos = []

        self.mu = 3000
        self.sigma = 300
        self.scale = 1800000

        self.debug = False

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

    def plot_reward(self,model_name,numruns,timediff):
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
        plt.savefig(f"model_v6.all/{model_name}_reward.png")
        print("\nsaved reward\n")
        # plt.show()
        # plt.close()

    def plot_time(self,model_name):
                
        print(f"{len(self.time_arr)/len(self.reward_arr)*100}% converged")
        print(f"{(1-(len(self.time_arr)/len(self.reward_arr)))*100}% failed")

        fig, ax = plt.subplots()

        ax.set_xlabel("time steps",fontsize=12)
        ax.set_ylabel("count",fontsize=12)
        ax.set_title(f"time steps for completion",fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)                

        try:
            ax.hist(self.time_arr,range = (min(self.time_arr),max(self.time_arr)),bins=25,log=True)
        except:
            print("No time values")

        plt.savefig(f"model_v6.all/{model_name}_time.png")
        print("\nsaved time\n")        
        # plt.show()
        # plt.close()


    def plot_diff(self,model_name):

        fig, ax = plt.subplots()

        ax.set_xlabel("distance", fontsize=12)
        ax.set_ylabel("count",fontsize=12)
        ax.set_title(f"Terminal position & fn max position difference",fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=8)        

        ax.hist(self.diff_arr,bins=25,log=True)
        plt.savefig(f"model_v6.all/{model_name}_diff.png")
        print("\nsaved diff\n")        
        # plt.show()
        # plt.close()

    def plot_final(self,model_name):
        
        fig, ax = plt.subplots()

        ax.set_xlabel("final position")
        ax.set_ylabel("count")
        ax.set_title(f"final state")        

        ax.hist(self.final_arr,bins=25,log=True)
        plt.savefig(f"model_v6.all/{model_name}_Lfinal.png")
        print("\nsaved final\n")        
        # plt.show()
        # plt.close()
        

    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2000
        yScale = 10/3500
        sScale = 10/200
        cScale = 10/600
        # cScale = 1
        for index in tiles(self.iht,16,[state[0]*yScale, state[1]*sScale, state[2]*cScale, act*aScale]):
        # for index in tiles(self.iht,16,[state[0]*zScale, state[1]*sScale, act*aScale]):        
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
            print(f"-->(z,action): ({state[0]},{action})")
            print(f"-->new z: {new_z}\n")

        terminated = False
        # if (abs(new_slope)<=1) and (new_curve>=250):
        if (abs(new_slope)<=1) and (self.f(new_z)>1500) and  (new_curve>=500):
            terminated = True
            reward = 10

        return [[self.f(new_z),new_slope,new_curve],reward,terminated]


    def episode(self,i):
        
        reward_sum = 0
        t = 0
        
        rng = np.random.default_rng()
        sigma = rng.normal(300,100)
        self.sigma = sigma if sigma>100 else 100
        mu = rng.normal(3200,300)
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

        while not terminated:
            
            if self.debug:
                print(f"---------------time step {t}---------------")                
            
            new_state, reward, terminated = self.step(state,action)
            new_action = self.policy(new_state)
            reward_sum += reward

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


        if converged:
            self.time_arr.append(t)            
            self.diff_arr.append(abs(self.zPos[-1]-self.mu))
        if not converged:
            self.final_arr.append(self.zPos[-1])            
        self.reward_arr.append(reward_sum)

            
def main():
    
    model_name = "v6.7_5e5"
    m = TOsarsa(model=f'model_v6.all/model_{model_name}.pickle')

    m.debug = False


    # fig, ax = plt.subplots()

    # ax.set_xlabel("element number")
    # ax.set_ylabel("value")
    # ax.set_title(f"Weight vector")
    
    # ax.plot(np.arange(0,len(m.weights)),m.weights,"-",marker='s',color="royalblue")
    # plt.show()
    # plt.close()

    
    m.end_time = 200
    numruns = 5000
    st = time.time()
    for i in range(int(numruns)):
        m.episode(i)

    et = time.time()

    m.plot_time(model_name)
    m.plot_reward(model_name,numruns,et-st)
    m.plot_diff(model_name)
    m.plot_final(model_name)


if __name__ == "__main__":
    main()
        




