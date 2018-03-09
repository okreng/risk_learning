Authors: Oliver Krengel, Henry Chen, Edward Terry
Carnegie Mellon University
10703: Deep Reinforcement Learning
Spring 2018


This project attempts to learn "Risk" (board game) strategy through reinforcement learning methods.  



--------------------- Coding and Data Management Standards ----------------------------

Directories: 
If multiple of the same basic file type/class begin to accumulate, put them in a subdirectory under the name <class> e.g. boards


Git: 
Commit often
Create new files, do not branch, if starting new agent or network classes
Branch if editing classes in-use
Merge only after testing
DO NOT DELETE ANY FILES - move to archives if code/data is no longer in use
.gitignore if adding large files (extended screen captures, etc)
UNANTICIPATED CONFLICTS - Retain local repository and post on Issues tab in github


Environment:
DO NOT DEVELOP IN OTHER ENVIRONMENTS

Required:
Python 3.5
Tensorflow 2.0
Keras 2.0 

Recommended:
Cuda 9.0, CuDNN 7.0


Python standards:
Include file headers for each unique file
Use tabs for spacing
Maximum 80 characters per line

Naming: 
module_name, package_name, ClassName, method_name, ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
global_var_name, instance_var_name, function_parameter_name, local_var_name

files:
Begin all words with lowercase, underscore between words, end in .py

Comments:
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


Spacing:
Two blank lines between top-level definitions, one blank line between method definitions.

Space between every argument
Maintain clarity with large equations

Exceptions:
Mark in-progress code as TODO above working section, as TODO is caught by pycharm 
e.g.

#TODO: change output nums ########
# output = 6
output = 7



------------------------Project Description----------------------------

The risk_env file defines two classes:
board - which creates a user-defined Risk board from a .risk file
env - the environment which defines the states, actions, rewards, and transition function for any given board

The agent_simple class defines an agent to perform a basic policy in an environment

