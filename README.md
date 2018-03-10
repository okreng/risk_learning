Authors: Oliver Krengel, Henry Chen, Edward Terry
Carnegie Mellon University
10703: Deep Reinforcement Learning
Spring 2018


This project attempts to learn "Risk" (board game) strategy through reinforcement learning methods.  



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

#TODO: change output nums ########
# output = 6
output = 7



------------------------Project Description----------------------------

##risk_graph.py
risk_graph file defines Four classes:
RiskGraph
Edge
EdgeSet
Territory - which contains the names of all edge territories, name, territory id, number of troops, and player id

##risk_game.py
risk_game file takes the graph and makes it into a game, adding an initialization, players, actions, and a win condition

##risk_env.py
risk_env takes the game and plays it to completion, given a board and player policies, it also provides rewards to players 


##.risk files
.risk files are kept in the boards directory
The file structure is as follows:
<territory_0_name>: <neighbor_1_name>, <neighbor_2_name>, ..., <neighbor_N_name>
<territory_1_name>: ....
<territory_T_name>: ...
Borders do not have to be defined in both directions, but it is recommended

##.mu files
.mu (matchup) files are kept in the matchups directory
The file structure is as follows:
<player_1_type>, <player_2_type>, ... <player_P_type>,
These are for specifying which policies are being run by the different players


##agent_simple.py
The agent_simple class defines an agent to perform a basic policy in an environment

