# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:58:33 2018

@author: gibra
"""

import math
import numpy as np

class Particle:
    "Create particles"

    def __init__(self,
                 permutacion,
                 insInfo):

        self.insInfo = insInfo

        # [123*456*...n], schedule.
        self.Schedule = permutacion

        # [raw material, time]
        self.Objetives = []

        # self.Tiempo = 0
        # self.Confianza = 0
        # self.Rojos = 0

        # amount of raw material
        self.q = insInfo.V

        # life time of the material
        self.sigma = insInfo.T

        # Containers number
        self.m = 1

        # start particle velocity
        self.Vel = np.random.randint(0,
                                     2,
                                     len(permutacion)).tolist()


###################################################################################################################
    # this function calculates the raw material cost (permutation [123*456*899... n]). EDPSO problem
    def CostoDeMaterial(self):
        # Machine index
        maquina = 0

        # raw material cost in that machine
        material = self.insInfo.instancia["Obj0" + str(maquina)].values.tolist()

        # time cost in that machine
        tiempo = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()

        for job in self.Schedule:
            if job == '*':
                maquina += 1
                material = self.insInfo.instancia["Obj0" + str(maquina)].values.tolist()
                tiempo = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()
            else:
                bk = material[int(job)]
                pk = tiempo[int(job)]

                if self.q >= bk and self.sigma > pk:
                    self.q -= bk
                    self.sigma -= pk
                elif self.q < bk and self.sigma > pk:
                    contenedores = math.ceil(bk / self.insInfo.V)
                    self.q = self.q - bk + contenedores * self.insInfo.V
                    self.sigma = self.insInfo.T - pk
                    self.m += contenedores
                elif self.q >= bk and self.sigma <= pk:
                    self.q = 0
                    self.sigma = 0
                elif self.q < bk and self.sigma <= pk:
                    if self.insInfo.T > pk:
                        contenedores = math.ceil(bk / self.insInfo.V)
                        self.q = self.q - bk + contenedores * self.insInfo.V
                        self.sigma = self.insInfo.T - pk
                        self.m += contenedores
                    else:
                        self.q = 0
                        self.sigma = 0

        self.Objetives.append(self.m * self.insInfo.V)


###################################################################################################################
    # this function calculates the time cost (permutation [123*456*899... n]) EDPSO problem
    def CostoProcesamiento(self):
        # machine
        maquina = 0

        # time cost in machine
        process = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()
        CostoProceso = 0

        for job in self.Schedule:
            if job == '*':
                maquina += 1
                process = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()
            else:
                chk = process[int(job)]
                CostoProceso += chk

        self.Objetives.append(CostoProceso)


###################################################################################################################
    # this function calculates the time cost (permutation [123*456*899... n]). Case study (f1)
    # Section 4.1 in paper. Equations 23.
    def CostoTiempo(self):
        # machine index
        maquina = 0

        # time cost in machine
        process = self.insInfo.instancia["Obj0" + str(maquina)].values.tolist()
        tiempoMaquina = 0
        tPm = []

        for job in self.Schedule:
            if job == '*':
                tPm.append(tiempoMaquina)
                tiempoMaquina = 0
                maquina += 1
                process = self.insInfo.instancia["Obj0" + str(maquina)].values.tolist()
            else:
                time = process[int(job)]
                tiempoMaquina += time

        costoT = max(tPm)
        self.Objetives.append(round(costoT))


###################################################################################################################
    # this function calculates the certainty (permutation [123*456*899... n]). Case study (-f2)
    # Section 4.1 in paper. Equations 26.
    def nivelConfianza(self):
        maquina = 0
        conf = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()
        nconf = []
        for job in self.Schedule:
            if job == '*':
                maquina += 1
                conf = self.insInfo.instancia["Obj1" + str(maquina)].values.tolist()
            else:
                nconf.append(conf[int(job)])

        confianza = np.median(nconf)
        self.Objetives.append(round(confianza))


###################################################################################################################
    # this function calculates the possibility of a task being interrupted (permutation [123*456*899... n]).
    #  Case study (f3) Section 4.1 in paper. Equation 27.
    def porcentajeRojos(self):
        maquina = 0
        crojo = self.insInfo.instancia["Obj2" + str(maquina)].values.tolist()
        projo = []

        for job in self.Schedule:
            if job == '*':
                maquina += 1
                crojo = self.insInfo.instancia["Obj2" + str(maquina)].values.tolist()
            else:
                projo.append(crojo[int(job)])

        porcentajeRojos = max(projo)
        self.Objetives.append(round(porcentajeRojos))


###################################################################################################################
    # Dominates concept by Pareto.
    def Dominates(self,
                  particula):
        dominates = True
        x = -1 * np.array(self.Objetives)
        y = -1 * np.array(particula.Objetives)
        o = 0
        for i in range(len(x)):
            if x[i] < y[i]:
                dominates = False
                break;
            elif x[i] == y[i]:
                o += 1

        if o == self.insInfo.obj:
            dominates = False

        return dominates


###################################################################################################################
    # Sigma Equation 4, Section 2.2 in paper.
    def getSigma(self,
                 particula):
        x = -1 * np.array(self.Objetives)
        y = -1 * np.array(particula.Objetives)

        c = self.concordancia(x, y)
        d = self.discordancia(x, y)

        sigma = c * d
        return sigma


###################################################################################################################
    # Strict preference (xPy) Table 1 in Section 2.2 in paper.
    def PE(self,
           sigmaXY,
           sigmaYX):
        preferencia = False

        if sigmaXY >= self.insInfo.lambdaE:
            if sigmaYX < 0.5:
                preferencia = True
            elif (0.5 <= sigmaYX < self.insInfo.lambdaE) and (sigmaXY - sigmaYX) >= self.insInfo.B:
                preferencia = True

        return preferencia


###################################################################################################################
    # Preference relation for (xQy, xRy, xKy) Table 1 in Section 2.2 in paper.
    def PD(self,
           sigmaXY,
           sigmaYX):
        preferencia = False

        if sigmaXY >= self.insInfo.lambdaE and sigmaXY >= sigmaYX:
            preferencia = True
        elif 0.5 <= sigmaXY < self.insInfo.lambdaE:
            preferencia = True
        elif sigmaYX < 0.5:
            preferencia = True
        elif (sigmaXY - sigmaYX) > (self.insInfo.B / 2):
            preferencia = True

        return preferencia


###################################################################################################################
    # The concordance index. Section 2.2 in paper. Equations 5.
    def concordancia(self,
                     x,
                     y):
        i = 0
        c = 0
        for ox, oy in zip(x, y):
            xIy = abs(ox - oy)
            if xIy <= self.insInfo.u[i]:
                c += self.insInfo.W[i]
            elif ox > oy:
                c += self.insInfo.W[i]

            i += 1
        return c


###################################################################################################################
    # The discordance index. Section 2.2 in paper. Equations 9.
    def discordancia(self,
                     x,
                     y):
        d = []
        i = 0
        for ox, oy in zip(x, y):
            delta = oy - ox
            if delta < self.insInfo.S[i]:
                d.append(1 - 0)
            elif self.insInfo.S[i] <= delta < self.insInfo.Veto[i]:
                valor = (delta - self.insInfo.u[i]) / (self.insInfo.Veto[i] - self.insInfo.u[i])
                d.append(1 - valor)
            elif delta >= self.insInfo.Veto[i]:
                d.append(1 - 1)

            i += 1
        return min(d)