# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:24:41 2019

@author: gibra
"""

import pandas as pd
import numpy as np
from Particle import Particle
from operator import itemgetter, attrgetter
from time import time
import numpy as np
import copy



class Algorithm:

    def __init__(self,
                 insInfo):
        self.insInfo = insInfo

        """
        arrays to store the population, nube is for all particles, paeretoSet is for the elitist particles and
        candidatos is for all particles in the final iteration
        """
        self.nube = []
        self.paretoSet = [[]]
        self.candidatos = [[]]


###################################################################################################################
    # here we create the cloud
    def generarNube(self):
        N = []
        # P1 technique for particle generation
        N1 = self.estrategiaN1()
        # P2 technique for particle generation
        N2 = self.estrategiaN2()
        # P3 technique for particle generation
        N3 = self.estrategiaN3()
        N = N1 + N2 + N3

        return N


###################################################################################################################
    def getParticleRandom(self):
        particle = np.random.permutation(self.insInfo.jobs).tolist()
        x = 0
        while (x < self.insInfo.m - 1):
            index = np.random.randint(1, self.insInfo.n - 1)
            if particle[index - 1] != "*" and particle[index + 1] != "*":
                particle.insert(index, "*")
                x += 1
        particle = self.SortJobs(particle)

        return particle


###################################################################################################################
    # this is the P1 technique. P1 randomly generates shcedules. Section 3 in paper
    def estrategiaN1(self):
        N1 = []
        for i in range(999 - self.insInfo.obj):
            particle = self.getParticleRandom()
            # particle instance
            P = Particle(particle,
                         self.insInfo)
            # first objective
            P.CostoDeMaterial()
            # second objective
            P.CostoProcesamiento()
            # P.CostoTiempo(P.Schedule)
            # P.nivelConfianza(P.Schedule)
            # P.porcentajeRojos(P.Schedule)
            N1.append(P)

        return N1


###################################################################################################################
    # this is the P2 technique. P2 uses the full-greedy technique. Section 3 in paper
    def estrategiaN2(self):
        N2 = []
        objetivos = []
        for o in range(self.insInfo.obj):
            ColumnsName = self.insInfo.instancia[["Obj" + str(o) + str(x) for x in range(self.insInfo.m)]].columns.values.tolist()
            # append objective values.
            objetivos.append(self.insInfo.instancia[ColumnsName].values)

        particle = []
        ul = self.insInfo.jobs.copy()
        pih = np.empty_like(self.insInfo.jobs)

        # full-greedy technique. Section 3 in paper. Algorithm 1.
        while ul:
            nk = []
            i = np.random.choice(ul)
            for k in range(self.insInfo.m):
                onk = 1
                for o in range(self.insInfo.obj):
                    onk *= objetivos[o][i][k]
                nk.append(onk)
            h = np.argmin(nk)
            pih[i] = h
            ul.remove(i)

        for k in range(self.insInfo.m - 1):
            maquina = np.where(pih == k)[0].tolist()
            maquina.append("*")
            particle += maquina
        maquina = np.where(pih == self.insInfo.m - 1)[0].tolist()
        particle += maquina

        # Sorting of jobs in ascending order based on time (SPT). Section 3 in paper
        particle = self.SortJobs(particle)
        P = Particle(particle,
                     self.insInfo)
        P.CostoDeMaterial()
        P.CostoProcesamiento()
        # P.CostoTiempo(P.Schedule)
        # P.nivelConfianza(P.Schedule)
        # P.porcentajeRojos(P.Schedule)
        N2.append(P)

        return N2


###################################################################################################################
    # this is the P3 technique. P3 uses the half-greedy technique. Section 3 in paper
    # m = number of machines
    # n = number of jobs
    def estrategiaN3(self):
        N3 = []
        objetivos = []
        particulas = []
        for o in range(self.insInfo.obj):
            ColumnsName = self.insInfo.instancia[["Obj" + str(o) + str(x) for x in range(self.insInfo.m)]].columns.values.tolist()
            objetivos.append(self.insInfo.instancia[ColumnsName].values)

        ul = self.insInfo.jobs.copy()
        pi = np.empty([self.insInfo.obj, self.insInfo.n]).tolist()

        # half-greedy technique. Section 3 in paper. Algorithm 2.
        while ul:
            i = np.random.choice(ul)
            for o in range(self.insInfo.obj):
                pi[o][i] = np.argmin(objetivos[o][i])
            ul.remove(i)

        for o in range(self.insInfo.obj):
            permu = []
            mObj = np.array(pi[o])
            for k in range(self.insInfo.m - 1):
                p = np.where(mObj == k)[0].tolist()
                p.append("*")
                permu += p
            permu += np.where(mObj == self.insInfo.m - 1)[0].tolist()
            # Sorting of jobs in ascending order based on time (SPT). Section 3 in paper
            particulas.append(self.SortJobs(permu))

        for par in particulas:
            P = Particle(par,
                         self.insInfo)
            P.CostoDeMaterial()
            P.CostoProcesamiento()
            # P.CostoTiempo(P.Schedule)
            # P.nivelConfianza(P.Schedule)
            # P.porcentajeRojos(P.Schedule)
            N3.append(P)

        return N3


###################################################################################################################
    # SPT strategy to sort the sequence of jobs in each machine. Section 3 in paper.
    # this function receives a particle as input.
    def SortJobs(self,
                 particle):
        # array of each machine index
        machineIndex = [i for i, x in enumerate(particle) if x == '*']
        j = 0
        part = []
        machineJobs = particle[:machineIndex[0]]
        times = np.array([t for t in self.insInfo.instancia.loc[machineJobs, "Obj1" + str(j)]])
        # times = np.array([t for t in instancia.loc[machineJobs,"Obj0"+str(j)]])
        jsorted = np.argsort(times)
        sortedJobs = [machineJobs[i] for i in jsorted]
        sortedJobs.append("*")
        part += sortedJobs

        # SPT cycle. Section 3 in paper.
        for i in range(len(machineIndex) - 1):
            j += 1
            start = machineIndex[i] + 1
            finish = machineIndex[i + 1]
            machineJobs = particle[start:finish]
            times = np.array([t for t in self.insInfo.instancia.loc[machineJobs, "Obj1" + str(j)]])
            # times = np.array([t for t in instancia.loc[machineJobs, "Obj0"+str(j)]])
            jsorted = np.argsort(times)
            sortedJobs = [machineJobs[p] for p in jsorted]
            sortedJobs.append("*")
            part += sortedJobs

        machineJobs = particle[machineIndex[-1] + 1:]
        times = np.array([t for t in self.insInfo.instancia.loc[machineJobs, "Obj1" + str(j + 1)]])
        # times = np.array([t for t in instancia.loc[machineJobs,"Obj0"+str(j+1)]])
        jsorted = np.argsort(times)
        sortedJobs = [machineJobs[i] for i in jsorted]
        part += sortedJobs

        return part


###################################################################################################################
    # Update the best known particle pBest. Section 3 in paper.
    def UpdatePbest(self,
                    Pbest):
        for i in range(len(self.nube)):
            # Pareto dominance concept
            dominatesN = self.nube[i].Dominates(Pbest[i])
            dominatesP = Pbest[i].Dominates(self.nube[i])

            # Equation 16. Page 15 in paper.
            if dominatesN is True:
                Pbest[i] = copy.deepcopy(self.nube[i])
            elif dominatesP is False:
                r = np.random.randint(1, 101)
                if r > 50:
                    Pbest[i] = copy.deepcopy(self.nube[i])

        return Pbest


###################################################################################################################
    # Update the best known global particle gBest. Section 3 in paper.
    def UpdateGbest(self,
                    paretoSet,
                    gBest):
        candidate = copy.deepcopy(np.random.choice(paretoSet))

        # Pareto dominance concept
        dominatescan = candidate.Dominates(gBest)
        dominatesgBest = gBest.Dominates(candidate)

        # Equation 17. Section 3 in paper.
        if dominatescan is True:
            return candidate
        elif dominatesgBest is False:
            r = np.random.randint(1, 101)
            if r > 50:
                return candidate

        return gBest


###################################################################################################################
    # the EDPSO technique to generate the elitist set, and generate the fronts.
    def ENS(self):
        size = int(self.insInfo.sizeP / 4)
        F = [[]]

        # Sorting of particles in ascending order based on first objective.
        nube = self.sortedNube(self.nube)

        # Sorting of particles in ascending order based on Pareto concept
        # generating fronts
        for particle in nube:
            f = self.Algorithm2(particle, F)
            if f > len(F) - 1:
                F.append([copy.deepcopy(particle)])
                temp = np.concatenate(F)
                if len(temp) > size:
                    F = self.CrowdingDistance(F)
            else:
                F[f].append(copy.deepcopy(particle))
                temp = np.concatenate(F)
                if len(temp) > size:
                    F = self.CrowdingDistance(F)

        return F


###################################################################################################################
    # delete same particles.
    def deleteSame(self,
                   F):
        aux = []
        F0 = []
        aux.append(F[0].Schedule)
        F0.append(F[0])
        if len(F) > 1:
            for i in range(1, len(F)):
                if F[i].Schedule not in aux:
                    aux.append(F[i].Schedule)
                    F0.append(F[i])

        return (F0)


###################################################################################################################
    # The O-PSO technique to generate the elitist set, and generate the fronts. Section 2 in paper. ELECTRE III method
    def ELECTRE(self,
                Nube):
        F = []
        Frente = []
        sigmas = []
        size = int(self.insInfo.sizeP / 4)
        preferencia = False

        # First step. Pareto dominance concept.
        for n in Nube:
            for part in Nube:
                preferencia = part.Dominates(n)
                if preferencia:
                    break

            if not preferencia:
                Frente.append(n)

        # Sigma calcuation. Equation 12, Section 2 in paper.
        # Outranking, concordance and discordance matrices are calculated here. Equations 3-12. Section 2 in paper
        for part in Frente:
            sigP = []
            for part2 in Frente:
                if part == part2:
                    sigP.append(1)
                else:
                    sigP.append(part.getSigma(part2))

            sigmas.append(sigP)

        # Sigma matrix, outraniking concept. Equation 12, Section 2 in paper.
        sigmas = np.array(sigmas)

        # Preference relation begins here. Section 2 in paper.

        # First we calculate the net-flow. Table 2, Section in paper.
        Fs = []
        for i in range(len(sigmas)):
            sXy = np.sum(sigmas[i])
            sYx = np.sum(sigmas[:, i])
            flujo = sXy - sYx
            Fs.append(flujo)

        # Vector of preference relations.

        Xsp = []
        for i in range(len(Frente)):
            xprima = []
            # solution not strictly outranking x, Table 2, Section 2 in paper. (Strict preference Table 1).
            ns = 0
            # Solutions weakly outranking x, Table 2, Section 2 in paper. (Weakpreference Table 1).
            nq = 0
            # Solutions that are greater than x in net flow-score
            fn = 0
            for j in range(len(Frente)):
                if i != j:
                    # preference relations. these methods are of the particula class
                    preE = Frente[j].PE(sigmas[j][i], sigmas[i][j])
                    preD = Frente[j].PD(sigmas[j][i], sigmas[i][j])
                    if preE:
                        ns += 1
                    if preD:
                        nq += 1
                    if Fs[j] > Fs[i]:
                        fn += 1

            xprima.append(ns)
            xprima.append(nq)
            xprima.append(fn)
            Xsp.append(xprima)

        # Remove solutions with strictly preference.
        for p, xs in zip(Frente, Xsp):
            if xs[0] > 0:
                Frente.remove(p)
                Xsp.remove(xs)

        # generating fronts
        # Problem Equation 13 Section 2 in paper.
        while len(Frente) != 0:
            fr = []
            xspremove = []
            tempf = Frente.copy()
            pref = False
            for i in range(len(tempf)):
                for j in range(len(tempf)):
                    if i != j:
                        # Pareto concept evaluation for the problem in equation 13.
                        pref = self.paretoFneto([Xsp[j][1],
                                            Xsp[j][2]],
                                           [Xsp[i][1],
                                            Xsp[i][2]])
                        if pref:
                            break

                if not pref:
                    fr.append(copy.deepcopy(tempf[i]))
                    Frente.remove(tempf[i])
                    xspremove.append(Xsp[i])

            if len(fr) == 0:
                fr = copy.deepcopy(Frente)
                Frente = []

            for k in xspremove:
                Xsp.remove(k)

            fr = self.deleteSame(fr)
            F.append(fr)

        aux = np.concatenate(F)
        while len(aux) > size:
            # function to eliminate excess solutions
            F = self.CrowdingDistance(F)
            aux = np.concatenate(F)

        return F


###################################################################################################################
    # This function returns the number of the front. EDPSO case.
    def Algorithm2(self,
                   particle,
                   F):
        x = len(F) - 1
        k = 0
        if not F[0]:
            return k
        else:
            while True:
                for i in range(len(F[k]) - 1, -1, -1):
                    existDom = F[k][i].Dominates(particle)
                    if existDom:
                        break;

                if not existDom:
                    return k
                    break
                else:
                    k += 1
                    if k > x:
                        return k
                        break


###################################################################################################################
    #  Sorting of particles in ascending order based on first objective. EDPSO case
    def sortedNube(self):
        objetivos = [i.Objetives for i in self.nube]
        temp = objetivos.copy()
        objetivos.sort(key=itemgetter(slice(0, None)))
        indices = [temp.index(o) for o in objetivos]
        nubeSorted = [self.nube[par] for par in indices]

        return nubeSorted


###################################################################################################################
    # This function eliminates particles by crowding distance if the elitist set is saturated.
    def CrowdingDistance(self,
                         F):
        Archive = np.concatenate(F).tolist()
        Aux = Archive.copy()
        n = len(Archive)
        distancias = np.zeros(n).tolist()

        for m in range(len(Archive[0].Objetives)):
            Aux = sorted(Aux, key=lambda e: e.Objetives[m])
            for i in range(1, n - 1):
                indice = Archive.index(Aux[i])
                distancias[indice] += (Aux[i + 1].Objetives[m] - Aux[i - 1].Objetives[m])

            distancias[Archive.index(Aux[0])] += np.max(distancias)
            distancias[Archive.index(Aux[-1])] += np.max(distancias)

        eliminate = Archive[np.argmin(distancias)]

        for p in range(len(F)):
            if eliminate in F[p]:
                if p == 0:
                    self.savePareto(eliminate)
                F[p].remove(eliminate)
                break

        return F


###################################################################################################################
    # this is local search algorithm. Section 3 in paper
    def lambdaSwap(self):
        for frente in self.paretoSet:
            for index in range(len(frente)):
                r = []
                while (len(r) < self.insInfo.lambdaS):
                    n = np.random.randint(len(frente[index].Schedule))
                    if n not in r:
                        r.append(n)

                vals = [frente[index].Schedule[r[i]] for i in range(-1, len(r) - 1, 1)]
                for i in range(len(r)):
                    frente[index].Schedule[r[i]] = vals[i]

                frente[index].Schedule = self.SortJobs(frente[index].Schedule)
                frente[index].Objetives = []
                frente[index].m = 1
                frente[index].sigma = self.insInfo.T
                frente[index].CostoDeMaterial()
                frente[index].CostoProcesamiento()
                # frente[index].CostoTiempo(frente[index].Schedule)
                # frente[index].nivelConfianza(frente[index].Schedule)
                # frente[index].porcentajeRojos(frente[index].Schedule)


###################################################################################################################
    # this is the function to update the operators. Section 3 in paper (Updating velocity and position of particles)
    def UpdatingOperators(self,
                          pBest,
                          gBest):
        for i in range(len(self.nube)):
            # the subtraction, selection, and-or operators
            delta1P = self.Substraction(self.nube[i].Schedule, pBest[i].Schedule)
            delta1G = self.Substraction(self.nube[i].Schedule, gBest.Schedule)
            deltakC1 = self.Select(delta1P, self.insInfo.c1)
            deltakC2 = self.Select(delta1G, self.insInfo.c2)
            deltakM = self.Select(self.nube[i].Vel, self.insInfo.M)
            self.nube[i].Vel = self.AndOr(deltakC1, deltakC2, deltakM)

            # Updating the position of the particles.
            self.nube[i].Schedule = self.SortJobs(self.MapOperator(self.nube[i].Schedule,
                                                                   self.nube[i].Vel))
            self.nube[i].Objetives = []
            self.nube[i].m = 1
            self.nube[i].sigma = self.insInfo.T
            self.nube[i].CostoDeMaterial()
            self.nube[i].CostoProcesamiento()
            # nube[i].CostoTiempo(nube[i].Schedule)
            # nube[i].nivelConfianza(nube[i].Schedule)
            # nube[i].porcentajeRojos(nube[i].Schedule)


###################################################################################################################
    # The subtraction operator (−). Section 3 in paper. Equations 19-22.
    def Substraction(self,
                     schedule1,
                     schedule2):
        delta1 = []
        for p1, p2 in zip(schedule1, schedule2):
            if p1 == p2:
                delta1.append(1)
            else:
                delta1.append(0)

        return delta1


###################################################################################################################
    # The selection operator (×). Section 3 in paper. Equation 23.
    def Select(self,
               delta,
               peso):
        deltak = []
        r = np.random.rand(len(delta)).tolist()
        for ran, i in zip(r, delta):
            if ran > peso:
                deltak.append(1 - i)
            else:
                deltak.append(i)

        return deltak


###################################################################################################################
    # The function of the and-or (+). Section 3 in paper. Equations 24-29.
    def AndOr(self,
              delta1,
              delta2,
              delta3):
        velocity = []
        r = np.random.rand(len(delta1)).tolist()
        for ran, d1, d2, d3 in zip(r, delta1, delta2, delta3):
            if ran < 0.5:
                if d1 == 0 or d2 == 0 or d3 == 0:
                    velocity.append(0)
                else:
                    velocity.append(1)
            else:
                if d1 == 1 or d2 == 1 or d3 == 1:
                    velocity.append(1)
                else:
                    velocity.append(0)

        return velocity


###################################################################################################################
    # Updating the position of the particles. Map operator in paper (Section 3). Equations 30-32.
    def MapOperator(self,
                    Schedule,
                    velocidad):
        per = np.zeros_like(Schedule).tolist()
        reorder = []
        for v, job in zip(velocidad, range(len(Schedule))):
            if v == 0:
                reorder.append(job)
            else:
                per[job] = Schedule[job]
        sper = np.random.permutation(reorder)

        for i, j in zip(reorder, sper):
            per[i] = Schedule[j]

        return per


###################################################################################################################
    # Function to create a txt with pareto particles.
    def savePareto(self,
                   P):
        f = open("ParetoArchive.txt", "a")
        for elemento in P.Schedule:
            f.write('%s ' % elemento)
        f.write('\n')

        for objetivo in P.Objetives:
            f.write('%s ' % objetivo)
        f.write('\n')


###################################################################################################################
    # Function to read the paretoarchive.
    def getParetoOptimal(self):
        ParetoSet = []
        file = open("ParetoArchive.txt", "r")
        Pars = file.readlines()
        for line in range(0, len(Pars), 2):
            p = list(map(self.enteros, Pars[line].split()))
            o = list(map(float, Pars[line + 1].split()))
            P = Particle(p)
            P.Objetives = o
            ParetoSet.append(P)
        file.close()

        return ParetoSet


###################################################################################################################
    # Function to update the paretoarchive.
    def reescribirParetoArchive(self,
                                Frente):
        f = open("ParetoArchive.txt", "w")
        for part in Frente:
            for dato in part.Schedule:
                f.write('%s ' % dato)
            f.write('\n')
            for dato in part.Objetives:
                f.write('%s ' % dato)
            f.write('\n')
        f.close()


###################################################################################################################
    # Function to delete particles in paretoarchive.
    def limpiarArchive(self):
        f = open("ParetoArchive.txt", "w")
        f.write('')
        f.close()


###################################################################################################################
    # Function to read solutions
    def enteros(self,
                n):
        if n != "*":
            return int(n)

        return n


###################################################################################################################
    # Pareto concept to evaluate net-flow. Equation 13 in paper. Section 2 in paper.
    def paretoFneto(self,
                    ox,
                    oy):
        Preferencia = True
        o = 0
        for i in range(len(ox)):
            if ox[i] > oy[i]:
                Preferencia = False
                break
            elif ox[i] == oy[i]:
                o += 1
        if o == self.insInfo.obj:
            Preferencia = False

        return Preferencia


###################################################################################################################
    def Execute(self):
        for exp in range(0, self.insInfo.MAX_EXPERIMENTS):
            print("Started experiment: ", exp)
            tiempoI = time()
            """
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            First we call the function generNube() to generate the point cloud. 
            Then, we call the ELECTRE function to obtain the best particles from this first cloud. 
            Next, we update pBest (local best) and gBest (global best). 
            Finally, we call lambdaSwap for the local search.
            """
            self.nube = self.generarNube()
            paretoNube = np.concatenate((self.nube, np.concatenate(self.paretoSet))).tolist()

            # preference method, outranking
            self.paretoSet = self.ELECTRE(paretoNube)
            pBest = copy.deepcopy(self.nube)
            gBest = copy.deepcopy(self.nube[0])

            # update global Best function
            gBest = self.UpdateGbest(self.paretoSet[0], gBest)

            self.lambdaSwap()
            """+++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

            """
            +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Here begins the optimization process, the algorithm will finish after 500 iterations. 
            First, we update the operators. 
            Then, we call the function Electre to obtain the elitist set. 
            Finally, we update the best particles and call the local search.
            """

            for i in range(self.insInfo.MAX_ITERATIONS):
                print("Iteration: ", i)

                # Function for the optimization process.
                self.UpdatingOperators(pBest,
                                       gBest)

                paretoNube = np.concatenate((self.nube,
                                             np.concatenate(self.paretoSet))).tolist()

                # preference method, outranking
                paretoSet = self.ELECTRE(paretoNube)

                pBest = self.UpdatePbest(pBest)

                gBest = self.UpdateGbest(paretoSet[0],
                                         gBest)
                # local search
                self.lambdaSwap()

            tiempoF = time()
            tiempoEjecucion = tiempoF - tiempoI
            print("Finished experiment: ", exp)
            print("Runtime: %.2f \n\n" % tiempoEjecucion)

            soluciones = np.concatenate((pBest, np.concatenate(self.paretoSet))).tolist()

            # preference method, outranking
            candidatos = self.ELECTRE(soluciones)

            # here we call the function to print the solutions found by OPSO and print them by fronts.
            for c, i in zip(candidatos, range(len(candidatos))):
                self.insInfo.saveFrente(c,
                                        i,
                                        tiempoEjecucion,
                                        exp)