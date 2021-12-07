# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:24:41 2019

@author: gibra
"""

import pandas as pd
import math
import os
import os.path


class InstanceInfo:

    def __init__(self,
                 initFile):
        # Get the current working directory
        cwd = os.getcwd()
        print("Current working directory: {0}".format(cwd))


        # Verify the existence of the .ini file.
        if  not os.path.exists(initFile):
            raise Exception(initFile + " file does not exist in current working directory.")


        # Open the .ini file
        with open(initFile) as f:
            lines = f.readlines()



        # Parse the instance parameters stored in the .ini file.
        dict = {}
        for line in lines:
            line = line.strip()
            if not line.startswith("#") and not line == "":
                line = line.replace("\n", "").replace("\t", "")
                tokens = line.split("=")
                parameter = tokens[0].strip()
                value = tokens[1].strip()
                dict[parameter] = value



        # Verify that all parameters exist in the .ini file.
        if not "InstanceName" in dict.keys():
            raise Exception("'InstanceName' parameter does not exist in " + initFile + " file.")

        if not "sizeP" in dict.keys():
            raise Exception("'sizeP' parameter does not exist in " + initFile + " file.")

        if not "obj" in dict.keys():
            raise Exception("'obj' parameter does not exist in " + initFile + " file.")

        if not "n" in dict.keys():
            raise Exception("'n' parameter does not exist in " + initFile + " file.")

        if not "m" in dict.keys():
            raise Exception("'m' parameter does not exist in " + initFile + " file.")

        if not "clase" in dict.keys():
            raise Exception("'clase' parameter does not exist in " + initFile + " file.")

        if not "T" in dict.keys():
            raise Exception("'T' parameter does not exist in " + initFile + " file.")

        if not "V" in dict.keys():
            raise Exception("'V' parameter does not exist in " + initFile + " file.")

        if not "M" in dict.keys():
            raise Exception("'M' parameter does not exist in " + initFile + " file.")

        if not "c1" in dict.keys():
            raise Exception("'c1' parameter does not exist in " + initFile + " file.")

        if not "c2" in dict.keys():
            raise Exception("'c2' parameter does not exist in " + initFile + " file.")

        if not "W" in dict.keys():
            raise Exception("'W' parameter does not exist in " + initFile + " file.")

        if not "e" in dict.keys():
            raise Exception("'e' parameter does not exist in " + initFile + " file.")

        if not "B" in dict.keys():
            raise Exception("'B' parameter does not exist in " + initFile + " file.")

        if not "lambdaE" in dict.keys():
            raise Exception("'lambdaE' parameter does not exist in " + initFile + " file.")

        if not "u" in dict.keys():
            raise Exception("'u' parameter does not exist in " + initFile + " file.")

        if not "S" in dict.keys():
            raise Exception("'S' parameter does not exist in " + initFile + " file.")

        if not "Veto" in dict.keys():
            raise Exception("'Veto' parameter does not exist in " + initFile + " file.")

        if not "MAX_EXPERIMENTS" in dict.keys():
            raise Exception("'MAX_EXPERIMENTS' parameter does not exist in " + initFile + " file.")

        if not "MAX_ITERATIONS" in dict.keys():
            raise Exception("'MAX_ITERATIONS' parameter does not exist in " + initFile + " file.")





        # Assign the value of the parameters to the variables of the class to propagate them to the algorithm functions.
        # instance name
        self.InstanceName = dict["InstanceName"]

        # cloud size
        self.sizeP = int(dict["sizeP"])

        # number of objectives
        self.obj = int(dict["obj"])

        # number of jobs
        self.n = int(dict["n"])

        # number of machines
        self.m = int(dict["m"])

        # class
        self.clase = int(dict["clase"])

        # life time of the material. EDPSO case
        self.T = int(dict["T"])

        # amount of raw material
        self.V = int(dict["V"])

        # Constants for velocity update
        self.M = float(dict["M"])
        self.c1 = float(dict["c1"])
        self.c2 = float(dict["c2"])

        # Elitist set size
        self.lambdaS = math.floor(math.log2(self.n))


        # Electre parameters raw material (EDPSO problem)
        # Weights
        self.W = list( map(float, dict["W"].split(",") ) )

        # epsilon
        self.e = float(dict["e"])

        # Beta
        self.B = float(dict["B"])

        # lambda for ELECTRE
        self.lambdaE = float(dict["lambdaE"])

        # Indifference threshold
        self.u = list( map(int, dict["u"].split(",") ) )

        # Pre-veto threshold
        self.S = list( map(int, dict["S"].split(",") ) )

        # Veto threshold
        self.Veto = list( map(int, dict["Veto"].split(",") ) )

        # Number of experiments
        self.MAX_EXPERIMENTS = int(dict["MAX_EXPERIMENTS"])

        # Number of iterations
        self.MAX_ITERATIONS = int(dict["MAX_ITERATIONS"])







        # Verify that the file containing the instance data exists.
        if  not os.path.exists(self.InstanceName):
            raise Exception(self.InstanceName + " file does not exist in current working directory or check 'InstanceName' parameter in " + initFile + " file.")



        # Read the instance data.
        self.instancia = pd.read_csv(self.InstanceName).drop("Unnamed: 0", 1)


        # we create array for jobs
        self.jobs = self.instancia["job"].values.tolist()


###################################################################################################################
    # Function to write solutions in txt
    def saveExperimento(self,
                        frente0,
                        tiempo,
                        prueba):
        f = open("experimentos/"+str(self.n)+"_"+str(self.m)+"/claseELECTRE"+str(self.clase)+"P"+str(prueba)+".txt","w")
        #f = open("experimentos/d_"+str(clase)+"m"+str(m)+"j"+str(n)+"-P"+str(prueba)+".txt","w")
        #f.write('Particulas Conjunto Elitista: ')
        #f.write('\n')
        #for particula in frente0:
         #   for dato in particula.Schedule:
          #      f.write('%s '%dato)
           # f.write('\n')
            #for dato in particula.Objetives:
             #   f.write('%s '%dato)
            #f.write('\n')
        #f.write('Particulas Conjunto Optimo: ')
        #f.write('\n')
        for particula in frente0:
            for dato in particula.Schedule:
                f.write('%s '%dato)
            f.write('\n')
            for dato in particula.Objetives:
                f.write('%s '%dato)
            f.write('\n')
        f.write('Tiempo de Ejecución: %.2f'%tiempo)
        f.close()


###################################################################################################################
    # Use this function
    # Function to save solutions in txt (Front array, index front, time execution, test)
    def saveFrente(self,
                   frente,
                   fr,
                   tiempo,
                   prueba):

        os.makedirs('experimentos/sinls/'+str(self.n)+"_"+str(self.m)+"/class"+str(self.clase), exist_ok=True)

        f = open("experimentos/sinls/"+str(self.n)+"_"+str(self.m)+"/class"+str(self.clase)+"/clase"+str(self.clase)+"m"+str(self.m)+"j"+str(self.n)+"-F"+str(fr)+"-P"+str(prueba)+".txt", "w")
        for particula in frente:
            for dato in particula.Schedule:
                f.write('%s ' % dato)
            f.write('\n')
            for dato in particula.Objetives:
                f.write('%s ' % dato)
            f.write('\n')
        f.write('Tiempo de Ejecución: %.2f' % tiempo)
        f.close()