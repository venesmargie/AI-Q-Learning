# AI-Q-Learning
Temporal Difference Q-learning

Description from COMP 4106, Winter 2021 with Matthew Holden, Carleton University

Description: Use temporal difference Q-learning to learn the optimal policy for a vacuum cleaner on 5 squares

![Screen Shot 2023-08-12 at 8 02 02 PM](https://github.com/venesmargie/AI-Q-Learning/assets/41605002/b863654a-77ef-4abe-8563-ead9550e3d3e)

Assume the vacuum cleaner starts in square 1. 
It can either:
1. Clean the current square
2. Move to a horizontal square
3. Move to a vetically adjacent square

Each square is either dirty or clean

The reward r associated with a state s is:
r(s) = -1 * number of dirty squares
Use the following parameters:
Gamma = 0.5 (discount factor)
Alpha = 0.1 (learning rate)

Initially estimate the Q-function as:
Q(s, a) = 0

A few important notes for your implementation:
1. Consider a single iteration of temporal difference Q-learning.
2. Use the provided simulated trajectories (which were generated under a random policy); do not generate new trajectories.

(Trajectory List is not in this repository)

Implementation: 

Pseudo Code
Description of Algorithms used

![Screen Shot 2023-08-12 at 8 10 03 PM](https://github.com/venesmargie/AI-Q-Learning/assets/41605002/55c4c14a-fab7-4507-801c-989b441622e4)

For calculating each q-value in each trajectory’s state and action, we implemented a global
nested q list dictionary where:

qList = {“state” : {“action”: qvalue} …. }

Key: state (string)
Value: dictionary
Key: action for that state (string = “C”, “D”, “U”, “L”, “R”)
Value: qValue calculation for that state and action

```{
for state in trajectory states:
  maxValue = [0]
  qSA = 0
  q = 0
  count = 0
  reward = state’s reward
  state = current state
  action = current action
  currentSquare = element[0]

  if we are in the second to the last element: #boundary case
  nextState = next state in the trajectory
  found = check_state_action(state, action) #checks if a given state’s action is a valid move
    if (found == False):
    return 0
    nextStateAction = next state action
    currentAction = current action
    if state is in our qList: #CASE 1: if the current state is in our qlist
      if nextState in qList: #is the next state is in our q list
      maxValue = next following states q values
      q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)

  if currentAction in qList: #CASE 2: if the current action is in our qlist and next action is in the qList
    if nextStateAction in qList:
      maxValue = next following states q values
      qSA = get q value for the state and action
      q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
      update_qlist(element, currentAction, q) #update the same action with new q value
      
    if currentAction not in qList[element]: #CASE 3: if current action is not in the qList, butnext action is
      if nextStateAction in qList.keys():
        maxValue = sT1_qList(nextState, nextStateAction) #calculates next trajectories values,
        gets the max values
        q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
        update_qlist(element, currentAction, q)
    elif ((element not in qList.keys()) and (nextStateAction in qList.keys())): #CASE 4: if current state is not in the list, but the next state is in the trajectory
      maxValue = sT1_qList(nextState, nextStateAction)
      q = qSA + alpha * (r + gamma*(max(maxValue)) - qSA)
      add_qlist(element, currentAction, q)
    else: #CASE 5: add a new state and action in the qList
      q = alpha * (r + max(maxValue) - 0) #since no qvalue pair yet, max value will be 0
      elementAction = data['Action'][count]
      add_qlist(element, elementAction, q)
      count+=1 #increment count to check if in the end of the list
}
```
