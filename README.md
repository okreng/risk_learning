Authors: Oliver Krengel, Henry Chen, Edward Terry
Carnegie Mellon University
10703: Deep Reinforcement Learning
Spring 2018


This project attempts to learn "Risk" (board game) strategy through reinforcement learning methods.  

## TODO: create territory:border dict in game and update policies to use as action space

--------------------- Coding and Data Management Standards ----------------------------

## Directories: 
If multiple of the same basic file type/class begin to accumulate, put them in a subdirectory under the name <class> e.g. boards


## Git: 
- Commit often
- Work only on branches, do not push to protected branches 
- Branch if editing classes in-use
- Merge only after testing
- DO NOT DELETE ANY FILES - move to archives if code/data is no longer in use
- .gitignore if adding large files (extended screen captures, etc)
- UNANTICIPATED CONFLICTS - Retain local repository and post on Issues tab in github


## Environment:
DO NOT DEVELOP IN OTHER ENVIRONMENTS

### Required:
- Python 3.5
-Tensorflow 2.0
- Keras 2.0 

### Recommended:
- Cuda 9.0, CuDNN 7.0


# Python standards:
Include file headers for each unique file
Use tabs for spacing
Maximum 80 characters per line

## Naming: 
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
global_var_name, instance_var_name, function_parameter_name, local_var_name

## files:
Begin all words with lowercase, underscore between words, end in .py

## Comments:
Use Pydocs to enable autogeneration of docs 
```[python]
class Foo(object):
    """
    Foo encapsulates a name and an age.
    """
    def __init__(self, name, age):
        """
        Construct a new 'Foo' object.

        :param name: The name of foo
        :param age: The ageof foo
        :return: returns nothing
        """
        self.name = name
        self.age = age
```


## Spacing:
Two blank lines between top-level definitions, one blank line between method definitions.

Space between every argument
Maintain clarity with large equations

## Exceptions:
Mark in-progress code as TODO above working section, as TODO is caught by pycharm 
e.g.
```[python]
#TODO: change output nums ########
# output = 6
output = 7
```


# Project Description

## risk_graph.py:
risk_graph file defines Four classes:
RiskGraph - The state of the board, as defined by the fully connected graph, territories with army numbers and player_ids
Edge - points at two connected territories and two connected nodes
EdgeSet - used to manage the edges so that they are not duplicated, can be accessed randomly, and point to territories.
Territory - which contains the names of all edge territories, name, territory id, number of troops, and player id

The purpose of the RiskGraph is to define the state of the board at any given time.  
This is a low-level, unabstracted representation that does not incorporate any actions.
TODO: Add continents - representation will need to be added to the .risk files
TODO: Order the graph (territories AND edges) in a consistent way that can be applied to many maps
TODO: Clean up the API, secure private member variables


## risk_game.py:
risk_game file defines Two classes:
RiskGame - The rules of playing the game, actions that can be taken by players (as functions of this class), initialization and win conditions
Player - Basic player implementation i.e. player_id, territories occupied, total_armies, isAlive, etc. 
         Also higher-level player attributes like policy and isAgent, though these are managed more by the environment.

risk_game file takes the graph and makes it into a game, adding an initialization, players, actions, and win conditions.
The distinction between this class and environment is very important.  
This class executes actions specified by the environment, updates the graph and effectively guarantees that our agent is actually playing risk.
Furthermore, using this class' functions will allow the environment to sample random states and train only one network at a time.  
If this is surprising, see below when the action spaces are discussed.
This class should also be able to execute env.gm.step() such that it will return the state immediately before the next agent action in all cases.
TODO: All the gameplay still needs to be defined, all I have done so far is initialize the game state. Note - assume optimal battle strategy as defined in papers in the slack group.
TODO: Clean up the API, secure private member variables.
TODO: Create agent-specific functions for each action that return the next state the agent regains control.

## risk_env.py:
this file contains only one class:
RiskEnv - This class manages the Markov Decision Processes of all agents acting in the environment.  
This is deliberately abstracted from the game itself because we will be attempting to use different representations of the states in order to represent the MDP.
This class owns the game class and commands it when to proceed and when to stop.
The agent(s) interface ONLY with this class (not lower-level classes) in order that the game appears to be an MDP to the agent.
TODO: Really getting into the meat of the project here: define state space efficiently, consistently, and in a way that facilitates training.
TODO: Create functions for agents to utilize the environment
TODO: Functions for model-based approaches if this is something we want to pursue


## Further classes .pymeariver:
TODO: Agents, recorders (we are recording EVERYTHING, including low-level game behavior), policies esp. Greedy, and obviously networks (I am hoping to use Tensorflow)

######### Note from Oliver 3/9-3/10 ########################
Please read the paper_raw.txt file if you want to see some of the other ideas I've had about how to approach this problem.
I'm sorry in particular about two things: class protection (I literally just don't know how to do it in Python, and didn't have the internet available to me)
                                          API incompleteness (I wanted to get the codebase up to a certain point, but I skipped a few important nuts and bolts)
I am conditionally sorry if this code ends up being shit.  As mentioned above, most of it was done w/o access to internet, which was a setback.
However, I hope it's clear what I'm trying to accomplish with the many levels of abstraction discussed above.
I am keen to discuss the responsibilities of our env class, game class, and agent class(es), and which functions fall where - because this was my first stopping point.
More TODO's in the code itself btw.

Other file types:
## .risk files
.risk files are kept in the boards directory
The file structure is as follows:
<territory_0_name>: <neighbor_1_name>, <neighbor_2_name>, ..., <neighbor_N_name>
<territory_1_name>: ....
<territory_T_name>: ...
Borders do not have to be defined in both directions, but it is recommended

## .mu files
.mu (matchup) files are kept in the matchups directory
The file structure is as follows:
<player_1_type>, <player_2_type>, ... <player_P_type>,
Player types are strings
These are for specifying which policies are being run by the different players


## Log directories
Whenever creating a new policy class which contains a network, create a log directory
called <module>.logs in the same folder as the module
Also create a 0.instance file with '0' as the only content
This is the head that other instance files will branch off of

## .instance files 
These files are made to hold unique ID's of models
.instance files are kept in log directories
These codify a tree structure for model growth
The .instance file is the base from which new models branch
Each file contains a single number that will be the unique ID of the next model built off that one
Newly constructed models built off existing models will have the structure:
c1-c2-c3 where c1 is the number from the tree base
These save a single number that indicates the number of models built off the tree head
