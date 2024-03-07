import numpy as np
from numpy.polynomial import Polynomial as P
from tiles3 import tiles, IHT
import pickle
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Button, Slider
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



class TOsarsa:
    def __init__(self,size=32768,epsilon=0.4,model="",model_name=''):

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
        self.numruns = 3000

        self.action_space = [-1000,-500,-100,-25,-5,-1,1,5,25,100,500,1000]
        self.zPos = []
        self.weight_indices = {i:0 for i in range(self.maxsize)}

        self.mu = 3000
        self.sigma = 300
        self.scale = 1800000

        self.debug = False

        self.name = model_name

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


    def weight_activation(self,end_time,numruns):

        self.end_time = end_time
        
        for i in range(numruns):
            self.episode(i)
            
        fig, ax = plt.subplots()

        ax.set_xlabel("element number")
        ax.set_ylabel("value")
        ax.set_title(f"model {self.name}")

        ax.vlines(np.arange(0,len(self.weights)),0,[self.weight_indices[i] for i in range(self.maxsize)],color="royalblue")    
        plt.show()
        plt.close()
        
        

    def weight_compare(self,model2):
        
 
        fig, ax = plt.subplots(1,2)

        ax[0].set_xlabel("element number")
        ax[0].set_ylabel("value")
        ax[0].set_title(f"model {self.name}")

        ax[1].set_xlabel("element number")
        ax[1].set_ylabel("value")
        ax[1].set_title(f"model {model2.name}")    

        ax[0].vlines(np.arange(0,len(self.weights)),0,self.weights,color="royalblue")    
        ax[1].vlines(np.arange(0,len(model2.weights)),0,model2.weights,color="royalblue")

        fig.tight_layout()
        plt.show()
        plt.close()

    def value_visualize(self,fq_val,act):
        
        fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"})        

        slope, curve = np.meshgrid(np.arange(-100,101),np.arange(0,601))

        # create vertices for a rotated mesh (3D rotation matrix)
        X =  slope 
        Y = curve

        for fq in fq_val:
            fq_vals =[]
            for x in range(len(X)):
                temp = []
                for y in range(len(Y[0])):        
                    temp.append(self.q((fq,X[x][y],Y[x][y]),act))
                fq_vals.append(temp)

            # for x in range(len(X)):
            #     for y in range(len(Y[0])):
            #         if fq_vals[x][y] >= 0:
            #             print(f"({X[x][y]}, {Y[x][y]}) fq_val: {fq_vals[x][y]} val: {self.q((fq,X[x][y],Y[x][y]),act)}")

            # show the 3D rotated projection
            cset = ax2.contourf(X, Y, fq_vals, 100, zdir='z', offset=fq, cmap=cm.plasma)

        ax2.set_zlim((0,3500))
        
        plt.colorbar(cset)
        plt.show()        
        


    def feature_fn(self,state,act):
        features = np.zeros(self.maxsize)
        aScale = 10/2000
        yScale = 10/3500
        sScale = 10/200
        cScale = 10/600
        # cScale = 1
        features = tiles(self.iht,16,[state[0]*yScale, state[1]*sScale, state[2]*cScale, act*aScale])       
        # for index in features:
        #     self.weight_indices[index] += 1
        #     features[index] = 1
        return features
        

    def q(self,state,action):
        
        sum = 0
        for index in self.feature_fn(state,action):
            sum += self.weights[index]
        return sum
        # return np.transpose(self.weights)@features
        

    
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
        if (abs(new_slope)<=1) and (self.f(new_z)>1500) and  (new_curve>=300):
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

        while not terminated:
            
            if self.debug:
                print(f"---------------time step {t}---------------")                
            
            new_state, reward, terminated = self.step(state,action)
            new_action = self.policy(new_state)
            reward_sum += reward

            if t == self.end_time:
                terminated = True

            if self.debug:
                print(f"state: {new_state}")
                print(f"action: {new_action}\n")
                
            state = new_state
            action = new_action

                
            if not terminated:
                t += 1

            
def main():
    
    model1_name = "v6.9_5e5"
    m = TOsarsa(model=f'model_v6.all/model_{model1_name}.pickle',model_name=model1_name)

    model2_name = "v6.7_5e5"
    n = TOsarsa(model=f'model_v6.all/model_{model2_name}.pickle',model_name=model2_name)    

    # st = time.time()
    # m.weight_activation(200,3000)
    # et = time.time()
    # print(f"time: {et-st:.2f} s")
    # n.weight_activation(200,3000)    
    # m.weight_compare(n)
    m.value_visualize([a for a in range(0,4000,1000)],-1)
    

    

if __name__ == "__main__":
    main()
        




