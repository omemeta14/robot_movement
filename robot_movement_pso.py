# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 00:50:19 2021

@author: Admin
"""


import math
import numpy as np
from numpy import linalg as LA
from math import radians as rad
from pyswarm import pso

l1 = l2 = l3 = 10

target = np.array([-15,-15])
prev_point = np.zeros(2,)
next_point = np.zeros(2,)

ang_deg_max = 20
n_step = 3
prev_step_count = 0

opt_action = np.zeros([n_step,3])

def robotarm_env(state,action,prev_step_count):
    
    n_step_episode = n_step
    
    t1_prev = state[0]
    t2_prev = t1_prev + state[1]
    t3_prev = t2_prev + state[2]
    #print(t1_prev,t2_prev,t3_prev)
    
    prev_point[0] = l1*math.cos(rad(t1_prev))+l2*math.cos(rad(t2_prev))+l3*math.cos(rad(t3_prev))
    prev_point[1] = l1*math.sin(rad(t1_prev))+l2*math.sin(rad(t2_prev))+l3*math.sin(rad(t3_prev))
    #print(prev_point)

    next_state = state + action
    #print(next_state)
    
    t1_next = next_state[0]
    t2_next = t1_next + next_state[1]
    t3_next = t2_next + next_state[2]   
    #print(t1_next,t2_next,t3_next)
    
    next_point[0] = l1*math.cos(rad(t1_next))+l2*math.cos(rad(t2_next))+l3*math.cos(rad(t3_next))
    next_point[1] = l1*math.sin(rad(t1_next))+l2*math.sin(rad(t2_next))+l3*math.sin(rad(t3_next))
    #print(next_point)
    
    reward = LA.norm(target-prev_point)-LA.norm(target-next_point)
    
    if (prev_step_count == n_step_episode-1):done = True

    else:done = False
    
    return reward,next_state,done


def objective_function(x):
    
    sum_reward = 0
    
    state = np.array([150,0,0])
    
    for prev_step_count in range(n_step):
    
        action = -ang_deg_max+x[3*prev_step_count:3*prev_step_count+3,]*(2*ang_deg_max)
        
        reward,next_state,done = robotarm_env(state,action,prev_step_count)

        sum_reward += reward
        
        state = next_state
    
    return -sum_reward

lb = np.zeros(n_step*3,)
ub = np.ones(n_step*3,)

xopt, fopt = pso(objective_function, lb, ub)

angopt = -ang_deg_max+xopt*(2*ang_deg_max)

# print('angopt, fopt = ', angopt,fopt)

for i in range(n_step):
    for j in range(3):
        opt_action[i][j] = angopt[j+(3*i)]

print("Robot arm movement")
print(opt_action)
print("Sum reward :",-fopt)

def robotarm_reward(state,action):
    
    t1_prev = state[0]
    t2_prev = t1_prev + state[1]
    t3_prev = t2_prev + state[2]
    # print(t1_prev,t2_prev,t3_prev)
    
    prev_point[0] = l1*math.cos(rad(t1_prev))+l2*math.cos(rad(t2_prev))+l3*math.cos(rad(t3_prev))
    prev_point[1] = l1*math.sin(rad(t1_prev))+l2*math.sin(rad(t2_prev))+l3*math.sin(rad(t3_prev))
    # print(prev_point)
    
    next_state = state + action
    
    t1_next = next_state[0]
    t2_next = t1_next + next_state[1]
    t3_next = t2_next + next_state[2]   
    # print(t1_next,t2_next,t3_next)
    
    next_point[0] = l1*math.cos(rad(t1_next))+l2*math.cos(rad(t2_next))+l3*math.cos(rad(t3_next))
    next_point[1] = l1*math.sin(rad(t1_next))+l2*math.sin(rad(t2_next))+l3*math.sin(rad(t3_next))
    # print(next_point)
    
    reward = LA.norm(target-prev_point)-LA.norm(target-next_point)
    
    return reward,next_state


# state = np.array([150,0,0])
# sum_reward = 0
# for i in range(len(opt_action)):
#     reward,next_state = robotarm_reward(state,opt_action[i,:])
#     state = next_state
#     sum_reward += reward
#     print("Reward :",reward)
# print("Sum Reward :",sum_reward)


