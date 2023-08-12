import sys
import numpy
import pandas as pd

class td_qlearning:

  alpha = 0.1
  gamma = 0.5

  qList = {}
  
  #constructor that should take one input argument
  def __init__(self, trajectory_filepath):
    # trajectory_filepath is the path to a file containing a trajectory through state space
    # Return nothing

    self.trajectory_filepath = trajectory_filepath
    data = get_column_data(trajectory_filepath)

    maxValue = [0] #store the max value in a list
    
    global qList
    qList = {}
    alpha = 0.1
    gamma = 0.5

    count = 0
    action = ""

    for element in data['State']:
      maxValue = [0] #array of the max value for the following trajectories 
      qSA = 0 #Q(S,A) value
      q = 0 #q value
      r = reward(element) #reward for that state
      state = data['State'][count]
      action = data['Action'][count]

      if (count < len(data) - 1):  #boundary: second to the last element or the last element 
        currentAction = data['Action'][count]
        nextState = data['State'][count + 1]
        nextStateAction = data['Action'][count + 1]
        
        found = check_state_action(state, action) #checks if a given state an action is a valid move
        if (found == False):
          return 0
        if element in qList.keys(): #if the element = state is in our qList 
          if nextState in qList.keys(): #if the next state exists in the q-list, use those values
            maxValue = sT1_qList(nextState, nextStateAction) #calculates next trajectories values, gets the max values
            q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
            
          #if our current action matches an action in our qlist
          if currentAction in qList[element]:
            if nextStateAction in qList.keys():
              maxValue = sT1_qList(nextState, nextStateAction) #calculates next trajectories values, gets the max values
            qSA = get_qlist(element, currentAction)
            q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
            update_qlist(element, currentAction, q)

          if currentAction not in qList[element]:
            if nextStateAction in qList.keys():
              maxValue = sT1_qList(nextState, nextStateAction) #calculates next trajectories values, gets the max values
            q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
            update_qlist(element, currentAction, q)

        elif ((element not in qList.keys()) and (nextStateAction in qList.keys())): #if current trajectory is not in the list, but the next states are in the trajectory 
          maxValue = sT1_qList(nextState, nextStateAction)
          q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
          add_qlist(element, currentAction, q)

        else:
          q = alpha * (r + max(maxValue) - 0) #since no qvalue pair yet, max value will be 0
          elementAction = data['Action'][count]
          add_qlist(element, elementAction, q)

        count+=1
  #getter trajectory file path
  def get_trajectory_filepath(self): 
    return self.trajectory_filepath

  #outputs the Q-value associated with that state-action pair
  def qvalue(self, state, action):
    # state s a string representation of a state
    # action is a string representation of an action
    #finalReward 
    q = 0
    if (state in qList and action in qList[state]):
      q = qList[state][action]
    return q

  #returns the optimal policy associated with this state
  def policy(self, state):

    # state is a string representation of a state
    l_val = -10
    r_val = -10
    d_val = -10
    u_val = -10

    l = "L"
    r = "R"
    c = "C"
    u = "U"
    d = "D"

    c_val = self.qvalue(state, c) 
    if (state in qList and check_state_action(state, l) == True):
        l_val = 0
    if (state in qList and check_state_action(state, r) == True):
        r_val = 0
    if (state in qList and check_state_action(state, d) == True):
        d_val = 0
    if (state in qList and check_state_action(state, u) == True):
        u_val = 0  
    if(check_state_action(state, l) != False and qList.get(state).get(l) != None):
        l_val = self.qvalue(state, l)    
    if(check_state_action(state, r) != False and qList.get(state).get(r) != None):
        r_val = self.qvalue(state, r)
    if(check_state_action(state, u) != False and qList.get(state).get(u) != None):
        u_val = self.qvalue(state, u)  
    if(check_state_action(state, d) != False and qList.get(state).get(d) != None):
        d_val = self.qvalue(state, d)
    # Return the optimal action under the learned policy
    optimal = max(l_val, r_val, c_val, u_val, d_val)

    if optimal == l_val:
        a = l 
    elif optimal == r_val:
        a = r
    elif optimal == c_val:
        a = c
    elif optimal == u_val:
        a = u
    elif optimal == d_val:
        a = d
 
    return a

def reward(state):
  reward = 0
  for letter in range(1, len(state)): 
    if state[letter] == '1':
      reward +=1 #add all dirty squares in our string
  reward*= -1 #multiply by -1
  return reward

def sT1_qList(state, action): #this returns the max values for a given st+1 trajectory
  maxValues = []

  if ("R" in qList[state]):
    maxValues.append(qList[state]["R"])

  if ("L" in qList[state]):
    maxValues.append(qList[state]["L"])

  if ("C" in qList[state]):
    maxValues.append(qList[state]["C"])

  if ("U" in qList[state]):
    maxValues.append(qList[state]["U"])

  if ("D" in qList[state]):
    maxValues.append(qList[state]["D"])
  
  #print("max Values: " + str(maxValues))
  return maxValues

def update_qlist(state, action, q): #this function updates q list with a given state and action pair
  qList[state][action] = q
  return

def add_qlist(state, action, q): #adds data to list
  global qList
  actionQValue = {action : q} #value of the state's key
  qList[state]= actionQValue #add to our dictionary 

def get_qlist(state, action): #returns q value of the state and action
  return qList[state][action] 

def check_state_action(state, action): #returns true or false if it is a valid move
  square = state[0]
  found = False

  if (square == "1"):
    if ("D" == action):
      found = True
    if ("C" == action):
      found = True

  if (square == "2"):
    if ("R" == action):
      found = True
    if ("C" == action):
      found = True

  if (square == "3"):
    if ("R" == action):
      found = True

    if ("L" == action):
      found = True

    if ("C" == action):
      found = True

    if ("U" == action):
      found = True

    if ("D" == action):
      found = True

  if (square == "4"):
    if ("L" == action):
      found = True
    if ("C" == action):
      found = True
  
  if (square == "5"):
    if ("C" == action):
      found = True

    if ("U" == action):
      found = True
  
  #print("max Values: " + str(maxValues))
  return found

def get_column_data(trajectory_filepath):
  column_data = pd.read_csv(trajectory_filepath, usecols=[0,1], names=['State', 'Action'])
  column_data = column_data.applymap(str) #converts map to a string 

  #example of how to get data 
  #print(column_data['State'])
  #print(column_data['Action'][0])
  #print((column_data['State'][1]))
  return column_data

def main():
  td_qlearning_instance = td_qlearning("trajectory.csv")
  global data 
  data = get_column_data(td_qlearning_instance.get_trajectory_filepath())

if __name__ == "__main__":
  main()
