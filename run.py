import gym, collections
import numpy as np
import sys
import plots
import settings
import copy
from random import uniform
from time import gmtime, strftime

from nn import NN


# MountainCar environment 
env = gym.make('MountainCar-v0')

# Number of runs experiment
NR = 100

# Number of episodes to observe in a single experiment
NE = 10000

# Max time steps in episode
T = 200
env.max_episode_steps = T

# prefix for files with policies diagrams
filePrefix = strftime("%Y%m%d_%H%M", gmtime())
  

"""
 Run single episode 
"""
def runEpisode(env, T, episodeNumber, render = False, randomPolicy = False):
    global allStates
 
    # All states in single episode
    episodeStates = []
    observation = env.reset()

    totalReward = 0
    
    if randomPolicy:
        """
        Take two random point from state spaces (p1,v1) (p2,v2). 
        This points define random decision boundry line, for one side of the line policy defines action push left (0) for the other push right (2). 
        No push action (0) is determined for states in 'width' distance from the line.
        """
        p1 = np.random.uniform(settings.pmin,settings.pmax)
        p2 = np.random.uniform(settings.pmin,settings.pmax)
        v1 = np.random.uniform(settings.vmin,settings.vmax)
        v2 = np.random.uniform(settings.vmin,settings.vmax)
        width = np.random.uniform(0,settings.wmax)
        
        for t in range(T):
            p = observation[0]    
            v = observation[1]    
            
            val = ((v-v1)*(p2-p1) - (v2-v1)*(p-p1))
            if val > width:
                action = 0
            elif val < -width:
                action = 2
            else:
                action = 1
            
            prevObservation = observation
            observation, reward, done, info = env.step(action)
            
            episodeStates.append([t,prevObservation, action])
    
            if done:
                if (t<T-1):
                    randomPolicy = []
                    X = np.linspace(settings.pmin, settings.pmax, settings.dS, endpoint=True)
                    Y = np.linspace(settings.vmin, settings.vmax, settings.dS, endpoint=True)
                    
                    for x in X:
                        for y in Y:
                            val = ((y-v1)*(p2-p1) - (v2-v1)*(x-p1))
                            if val > width:
                                randomPolicy.append([x,y,0])
                            elif val < -width:
                                randomPolicy.append([x,y,2])
                            else:
                                randomPolicy.append([x,y,1])
                    
                    #plots.plotPolicy(policy=randomPolicy,steps=t,points=[[p1,v1],[p2,v2]])
                    #plots.spaceDivision(point1=[p1,v1],point2=[p2,v2],xRange=[-1.2,0.6],yRange=[-0.07,0.07],width=width,steps=t)
                    
                return {'t':t, 'eps': episodeStates, 'policy': randomPolicy}
    
    else:
        for t in range(T):
            observation = np.append(observation, observation[0]*observation[1])
            action = np.argmax(nn.ask([observation]))
                
            prevObservation = observation
            observation, reward, done, info = env.step(action)
            
            if render:
                env.render()
            
            episodeStates.append([t,prevObservation, action, reward])
            
            if done:
                return {'t':t, 'eps': episodeStates}


    

"""
 Run NR experiments
"""
for e in range(1,NR):
    # Neural Network to store knowledge 
    nn = NN([3,30,15,3])
    
    mean = 0
    maxi = 0
    mint = 0 
    success = 0
    bestPolicies = []
    
    # In one run try NE episodes
    for i in range(1,NE):
        #  until achieve success randomNR times try random policy
        if success < settings.randomNR:
            res = runEpisode(env, T, i, render=False, randomPolicy=True)
            # remember random policy if it achieved the goal in 190 steps
            if (res['t'] < T-10 ):
                success += 1
                if success > 1:
                    bestPolicies = np.concatenate((bestPolicies, res['policy']), axis=0)
                else:
                    bestPolicies = res['policy']
                print ("Try {} completed after {} steps. {}/{} successful push policy with random decision boundry.".format(i,res['t'],success,settings.randomNR))
            continue
        # train NN after achieve randomNR success
        elif success == settings.randomNR:
            success += 1
            X =  [ [row[0],row[1],row[0]*row[1]] for row in bestPolicies]
            Y_ = [ row[2]          for row in bestPolicies]    
            
            Y = []
            for y in Y_:
                if y == 0:
                    Y.append([1,0,0])
                elif y == 1: 
                    Y.append([0,1,0])
                elif y == 2:
                    Y.append([0,0,1])
            
            for j in range (1,settings.learnN):
                Xn = copy.deepcopy(X)
                if j > 1: #add some random noise
                    print ("Train NN {}/{} with successful policies and some random noise.".format(j,settings.learnN))
                    for k in range (0,len(Xn)):
                        Xn[k][0] += uniform(-settings.maxRandomNoise, settings.maxRandomNoise)
                        Xn[k][1] += uniform(-settings.maxRandomNoise, settings.maxRandomNoise)
                        Xn[k][2] = Xn[k][0]*Xn[k][1] 
                else:
                    print ("Train NN {}/{} with successful policies.".format(j,settings.learnN))
                
                pTest = settings.dS ** 2
                pAll = len(X)
                
                for batch in range (0,pAll-pTest,100):
                    nn.train(Xn[batch:batch+100],Y[batch:batch+100])
                
                plots.plotPolicyNN(nn, saveToFile="{}_learn_step_{}".format(filePrefix,j))
                
            acu = nn.test(X[pAll-pTest:pAll],Y[pAll-pTest:pAll])
            print ("Accuracy",acu)
                
            plots.plotPolicyNN(nn,saveToFile="{}_endPolicy".format(filePrefix))
    

        res = runEpisode(env, T, i, render=True, randomPolicy=False)
        print ("Try {} done after {} steps.".format(i,res['t']))

