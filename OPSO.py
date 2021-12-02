# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:20:16 2018

@author: gibra
"""

# Here we import the classes to execute the algorithm

from InstanceInfo import InstanceInfo
from Algorithm import Algorithm





# here begins the execution of OPSO

if __name__ == "__main__":

    # Loads the parameters and data of the instance.
    insInfo = InstanceInfo("OPSO.ini")


    # Pass the instance data to the algorithm.
    alg = Algorithm( insInfo )
    alg.Execute()