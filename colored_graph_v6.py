#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkit as nk
import undirected_colored_graph as und
import math
import random
import sys
from collections import Counter

DEFAULT_NULL_COLOR = -1
DEFAULT_PAYOFF = 0
NULL_VERTEX = -1
from progress.bar import IncrementalBar
from networkit import graphtools


class coloredGraph:
    def __init__(self, g=None, kolors=None):
        self.__graph = g
        self.__colors = [DEFAULT_NULL_COLOR for i in range(self.__graph.numberOfNodes())]
        self.__payoff = [DEFAULT_PAYOFF for i in range(self.__graph.numberOfNodes())]
        self.__available = kolors
        self.__not_lemma_valid = False
        self.__LLL_THRESHOLD = 0
        self.__maxoutdeg = max([self.__graph.degreeOut(i) for i in range(self.__graph.numberOfNodes())])
        self.__maxindeg = max([self.__graph.degreeIn(i) for i in range(self.__graph.numberOfNodes())])
        self.__nu_nodes = set()
        self.__gammanu_nodes = set()
        self.__recompute_status = [1 for i in range(self.__graph.numberOfNodes())]

    def reset(self):
        self.__colors = [DEFAULT_NULL_COLOR for i in range(self.__graph.numberOfNodes())]
        self.__payoff = [DEFAULT_PAYOFF for i in range(self.__graph.numberOfNodes())]
        self.__LLL_THRESHOLD = 0
        self.__not_lemma_valid = False
        self.__nu_nodes = set()
        self.__gammanu_nodes = set()
        self.__recompute_status = [1 for i in range(self.__graph.numberOfNodes())]

    def getGraph(self):
        return self.__graph

    def getOrder(self):
        return self.__graph.numberOfNodes()

    def getSize(self):
        return self.__graph.numberOfEdges()

    def getColors(self):
        return self.__colors

    def getPayoffs(self):
        return self.__payoff

    def getAvailable(self):
        return self.__available

    def setAvailable(self, arr):
        self.__available = arr;

    def getNotLemmaValid(self):
        return self.__not_lemma_valid;

    def getLLLThreshold(self):
        return self.__LLL_THRESHOLD;

    def setLLLThreshold(self, threshold):
        self.__LLL_THRESHOLD = threshold

    def __computePayoffs(self):
        for i in range(self.__graph.numberOfNodes()):
            self.__updatePayoff(i)

    def __updatePayoff(self, vertex):

        if self.__graph.degreeOut(vertex) == 0:
            assert self.__payoff[vertex] == DEFAULT_PAYOFF
            # local gamma is one by definition, continue
            return;

        self.__payoff[vertex] = DEFAULT_PAYOFF
        for n in self.__graph.iterNeighbors(vertex):
            if self.__colors[vertex] != self.__colors[n]:
                self.__payoff[vertex] += 1;

    def getNEUnhappyVertex(self):
        unhappy_node = random.choice(list(self.__nu_nodes))
        self.__nu_nodes.remove(unhappy_node)
        return unhappy_node
        # return self.__nu_nodes.pop()

    def getGammaUnhappyVertex(self):
        unhappy_node = random.choice(list(self.__gammanu_nodes))
        self.__gammanu_nodes.remove(unhappy_node)
        return unhappy_node
        # return self.__gammanu_nodes.pop()

    def lenNUNodes(self):
        return len(self.__nu_nodes)

    def lenGammaUNodes(self):
        return len(self.__gammanu_nodes)

    def resetNodesStatus(self):
        self.__recompute_status = [1 for i in range(self.__graph.numberOfNodes())]

    def colored(self):
        return self.__getFirstNonColoredVertex() == NULL_VERTEX;

    def __getFirstNonColoredVertex(self):
        i = 0;

        while i < self.__graph.numberOfNodes():
            if self.__colors[i] == DEFAULT_NULL_COLOR:
                # self.__lastchecked = i;
                return i;
            i += 1;
        return NULL_VERTEX;

    def __getNonColoredNeighbor(self, vertex):
        for n in self.__graph.iterNeighbors(vertex):
            if self.__colors[n] == DEFAULT_NULL_COLOR:
                return n;
        return NULL_VERTEX;

    def numberOfUsedColors(self):
        return len(self.__listOfUsedColors());

    def __listOfUsedColors(self):
        return Counter(self.__colors).keys()

    def nashStatus(self):
        num_unhappy_NE = 0;
        num_unhappy_gamma_NE = 0;
        global_gamma = 1.0;
        is_NE = True;
        is_gamma_NE = True;
        for i in range(self.__graph.numberOfNodes()):
            if self.__graph.degreeOut(i) == 0 or (self.__graph.degreeOut(i) == 1 and self.__graph.degreeIn(i) == 0):
                continue;
            min_freq_NE, cset_NE = self.__isVertexNEUnhappy(i);
            min_freq_gamma_NE, cset_gamma_NE = self.__isVertexGammaNEUnhappy(i);
            if min_freq_gamma_NE != NULL_VERTEX:
                is_gamma_NE = False;
                num_unhappy_gamma_NE += 1;
                assert min_freq_gamma_NE != NULL_VERTEX;
                assert self.__graph.degreeOut(i) > 0;
                assert self.__colors[i] != DEFAULT_NULL_COLOR;
                assert self.__colors[i] not in cset_gamma_NE;
                assert (self.__graph.degreeOut(i) - min_freq_NE) >= self.__payoff[i];

                if self.__payoff[i] == DEFAULT_PAYOFF:  # all neighbors have same color of i
                    global_gamma = sys.maxsize  # default infinite gamma
                    if __debug__:
                        for nei in self.__graph.iterNeighbors(i):
                            assert self.__colors[i] == self.__colors[nei]
                else:
                    assert (self.__graph.degreeOut(i) - min_freq_NE) / self.__payoff[i] >= 1.0
                    global_gamma = max((self.__graph.degreeOut(i) - min_freq_NE) / self.__payoff[i], global_gamma)

            if min_freq_NE != NULL_VERTEX:
                is_NE = False;
                num_unhappy_NE += 1;
                assert min_freq_NE != NULL_VERTEX;
                assert self.__graph.degreeOut(i) > 0;
                assert self.__colors[i] != DEFAULT_NULL_COLOR;
                assert self.__colors[i] not in cset_NE;
                assert (self.__graph.degreeOut(i) - min_freq_NE) >= self.__payoff[i];

        if global_gamma == 1 and is_gamma_NE == False:
            raise Exception('mismatch gamma vs gamma_nash')
        if global_gamma > 1 and is_gamma_NE == True:
            raise Exception('mismatch gamma vs gamma_nash')
        assert num_unhappy_gamma_NE / self.__graph.numberOfNodes() >= 0 and num_unhappy_gamma_NE / self.__graph.numberOfNodes() <= 1

        if global_gamma == 1 and is_NE == False:
            for i in range(self.__graph.numberOfNodes()):
                if self.__graph.degreeOut(i) == 0 or (self.__graph.degreeOut(i) == 1 and self.__graph.degreeIn(i) == 0):
                    continue;
                assert (self.__payoff[i] != DEFAULT_PAYOFF);
                global_gamma = max((self.__graph.degreeOut(i) - min_freq_NE) / self.__payoff[i], global_gamma)

        if global_gamma > 10000:
            self.__computePayoffs()
            for v in self.__graph.iterNodes():
                if self.__graph.degreeOut(v) == 0 or (self.__graph.degreeOut(v) == 1 and self.__graph.degreeIn(v) == 0):
                    continue;
                if self.__payoff[i] == DEFAULT_PAYOFF:
                    for nei in self.__graph.iterNeighbors(v):
                        if self.__colors[v] != self.__colors[nei]:
                            raise Exception('Incoerente')

        return global_gamma, is_NE, (num_unhappy_NE / self.__graph.numberOfNodes()), \
               is_gamma_NE, (num_unhappy_gamma_NE / self.__graph.numberOfNodes());

    def efficientNashStatus(self):
        global_gamma = 1.0

        for i in range(self.__graph.numberOfNodes()):
            if self.__recompute_status[i] == 0:
                continue
            self.__recompute_status[i] = 0
            if self.__graph.degreeOut(i) == 0 or (self.__graph.degreeOut(i) == 1 and self.__graph.degreeIn(i) == 0):
                continue;
            min_freq_NE, cset_NE = self.__isVertexNEUnhappy(i)
            min_freq_gamma_NE, cset_gamma_NE = self.__isVertexGammaNEUnhappy(i)
            if min_freq_gamma_NE != NULL_VERTEX:
                self.__gammanu_nodes.add(i)
                if self.__payoff[i] == DEFAULT_PAYOFF:  # all neighbors have same color of i
                    global_gamma = sys.maxsize  # default infinite gamma
                else:
                    assert (self.__graph.degreeOut(i) - min_freq_gamma_NE) / self.__payoff[i] >= 1.0
                    global_gamma = max((self.__graph.degreeOut(i) - min_freq_gamma_NE) / self.__payoff[i],
                                       global_gamma)
            elif i in self.__gammanu_nodes:
                self.__gammanu_nodes.remove(i)

            if min_freq_NE != NULL_VERTEX:
                self.__nu_nodes.add(i)
            elif i in self.__nu_nodes:
                self.__nu_nodes.remove(i)
        for i in list(self.__gammanu_nodes):
            min_freq_gamma_NE, cset_gamma_NE = self.__isVertexGammaNEUnhappy(i)
            if self.__payoff[i] == DEFAULT_PAYOFF:  # all neighbors have same color of i
                global_gamma = sys.maxsize  # default infinite gamma
            else:
                assert (self.__graph.degreeOut(i) - min_freq_gamma_NE) / self.__payoff[i] >= 1.0
                global_gamma = max((self.__graph.degreeOut(i) - min_freq_gamma_NE) / self.__payoff[i],
                                   global_gamma)
        if global_gamma == 1 and len(self.__gammanu_nodes) > 0:
            raise Exception('mismatch gamma vs gamma_nash')
        if global_gamma > 1 and len(self.__gammanu_nodes) == 0:
            raise Exception('mismatch gamma vs gamma_nash')
        assert len(self.__gammanu_nodes) / self.__graph.numberOfNodes() >= 0 and \
               len(self.__gammanu_nodes) / self.__graph.numberOfNodes() <= 1

        if global_gamma == 1 and len(self.__nu_nodes) > 0:
            for i in range(self.__graph.numberOfNodes()):
                if self.__graph.degreeOut(i) == 0 or (self.__graph.degreeOut(i) == 1 and self.__graph.degreeIn(i) == 0):
                    continue;
                assert (self.__payoff[i] != DEFAULT_PAYOFF);
                global_gamma = max((self.__graph.degreeOut(i) - min_freq_NE) / self.__payoff[i], global_gamma)

        return global_gamma, len(self.__nu_nodes) == 0, (len(self.__nu_nodes) / self.__graph.numberOfNodes()), \
               len(self.__gammanu_nodes) == 0, (len(self.__gammanu_nodes) / self.__graph.numberOfNodes());

    def __isVertexNEUnhappy(self, vertex):
        # returns pair min_frequency, colorset (reason of unhappiness, or null)
        if self.__graph.degreeOut(vertex) == 0 or (
                self.__graph.degreeOut(vertex) == 1 and self.__graph.degreeIn(vertex) == 0):
            # outdegree null, always happy
            return [NULL_VERTEX, []];

        assert self.__colors[vertex] != DEFAULT_NULL_COLOR
        min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(vertex)

        if self.__colors[vertex] not in colorset:
            return [min_freq, colorset];

        return [NULL_VERTEX, []];

    def __isVertexGammaNEUnhappy(self, vertex):
        # returns pair min_frequency, colorset (reason of unhappiness, or null)
        if self.__graph.degreeOut(vertex) == 0 or (
                self.__graph.degreeOut(vertex) == 1 and self.__graph.degreeIn(vertex) == 0):
            # outdegree null, always happy
            return [NULL_VERTEX, []];

        assert self.__colors[vertex] != DEFAULT_NULL_COLOR
        min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(vertex)

        if self.__colors[vertex] not in colorset and self.__LLL_THRESHOLD * self.__payoff[vertex] < \
                self.__graph.degreeOut(vertex) - min_freq:
            return [min_freq, colorset];

        return [NULL_VERTEX, []];

    def randomColoring(self):
        self.__colors = [random.choice(self.__available) for i in range(self.__graph.numberOfNodes())];
        self.__computePayoffs();

    def __getRandomNEUnhappyVertex(self):
        # random triple unhappy vertex and reason if any (minfrequency and colorset) or null
        unhappy_vertices = []
        for i in range(self.__graph.numberOfNodes()):
            mfeq, colset = self.__isVertexNEUnhappy(i)
            if mfeq != NULL_VERTEX:
                unhappy_vertices.append([i, mfeq, colset])

        if len(unhappy_vertices) == 0:
            # assert self.isGammaNash(1)
            return [NULL_VERTEX, NULL_VERTEX, []];  # no unhappy in the graph

        # RANDOMLY ONE OF THE AVAILABLE UNHAPPY VERTICES
        return random.choice(unhappy_vertices), len(unhappy_vertices);

    def getNEUnhappyVertexInfos(self, i):
        mfeq, colset = self.__isVertexNEUnhappy(i)
        assert len(colset) > 0
        return i, mfeq, colset

    def getGammaUnhappyVertexInfos(self, i):
        mfeq, colset = self.__isVertexGammaNEUnhappy(i)
        return i, mfeq, colset

    def getNEUnhappyVertices(self):
        unhappy_vertices = []
        for i in range(self.__graph.numberOfNodes()):
            mfeq, colset = self.__isVertexNEUnhappy(i)
            if mfeq != NULL_VERTEX:
                unhappy_vertices.append([i, mfeq, colset])

        if len(unhappy_vertices) == 0:
            # assert self.isGammaNash(1)
            return []  # no unhappy in the graph

        # RANDOMLY ONE OF THE AVAILABLE UNHAPPY VERTICES
        return unhappy_vertices

    def __getRandomGammaUnhappyVertex(self):
        # random triple unhappy vertex and reason if any (minfrequency and colorset) or null
        unhappy_vertices = []
        for i in range(self.__graph.numberOfNodes()):
            mfeq, colset = self.__isVertexGammaNEUnhappy(i)
            if mfeq != NULL_VERTEX:
                unhappy_vertices.append([i, mfeq, colset])

        if len(unhappy_vertices) == 0:
            # assert self.isGammaNash(1)
            return [NULL_VERTEX, NULL_VERTEX, []], len(unhappy_vertices);  # no unhappy in the graph

        # RANDOMLY ONE OF THE AVAILABLE UNHAPPY VERTICES
        return random.choice(unhappy_vertices), len(unhappy_vertices);

    def __colorsWithMinimumFrequencyInNeighborhood(self, vertex):

        base_frequencies = {}
        for i in self.__available:
            base_frequencies[i] = 0;

        for n in self.__graph.iterNeighbors(vertex):
            if self.__colors[n] != DEFAULT_NULL_COLOR:
                base_frequencies[self.__colors[n]] += 1;

        min_f_color = min(base_frequencies.keys(), key=(lambda k: base_frequencies[k]))
        min_frequency_colors = []
        for k in base_frequencies.keys():
            if base_frequencies[k] == base_frequencies[min_f_color]:
                min_frequency_colors.append(k)

        return base_frequencies[min_f_color], min_frequency_colors;

    def improve(self):

        # CHANGE RANDOMLY ONE OF THE AVAILABLE UNHAPPY VERTICES
        unhappy_info, unhappy_num = self.__getRandomNEUnhappyVertex()
        vertex_to_change, minimumfrequency, colorset = unhappy_info
        if __debug__:
            assert self.__isVertexNEUnhappy(vertex_to_change)[0] != NULL_VERTEX
            assert self.__colors[vertex_to_change] not in colorset
            old_payoff = self.__payoff[vertex_to_change]

        self.__colors[vertex_to_change] = random.choice(colorset)
        self.__computePayoffs()

        if __debug__:
            # TEST IMPROVEMENT
            assert self.__payoff[vertex_to_change] == (self.__graph.degreeOut(vertex_to_change) - minimumfrequency)
            assert old_payoff <= self.__payoff[vertex_to_change]

    def improve_vertex(self, v_infos):

        # CHANGE RANDOMLY ONE OF THE AVAILABLE UNHAPPY VERTICES
        vertex_to_change, minimumfrequency, colorset = v_infos
        if __debug__:
            assert self.__isVertexNEUnhappy(vertex_to_change)[0] != NULL_VERTEX
            assert self.__colors[vertex_to_change] not in colorset
            old_payoff = self.__payoff[vertex_to_change]

        self.__colors[vertex_to_change] = random.choice(colorset)
        # self.__computePayoffs()
        for n in self.__graph.iterInNeighbors(vertex_to_change):
            self.__updatePayoff(n)
            self.__recompute_status[n] = 1
        self.__updatePayoff(vertex_to_change)
        self.__recompute_status[vertex_to_change] = 1

        if __debug__:
            # TEST IMPROVEMENT
            assert self.__payoff[vertex_to_change] == (self.__graph.degreeOut(vertex_to_change) - minimumfrequency)
            assert old_payoff <= self.__payoff[vertex_to_change]

    def getAverageGamma(self):
        sum_gamma = 0.0;
        for i in range(self.__graph.numberOfNodes()):
            if self.__isVertexNEUnhappy(i)[0] == NULL_VERTEX:
                sum_gamma += 1.;
                continue;

            else:
                assert self.__isVertexNEUnhappy(i)[0] != NULL_VERTEX;
                assert self.__graph.degreeOut(i) > 0;
                assert self.__colors[i] != DEFAULT_NULL_COLOR
                min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(i)
                # payoff,colorset = self.__colorsDifferentThanCurrentThatInduceMaximumPayoff(i)
                # print(i,self.__colors[i],colorset)
                assert self.__colors[i] not in colorset;
                # print(self.__graph.degreeOut(i),min_freq, self.__payoff[i])

                assert (self.__graph.degreeOut(i) - min_freq) >= self.__payoff[i];
                if self.__payoff[i] == DEFAULT_PAYOFF:  # all neighbors have same color of i
                    sum_gamma += (self.__graph.degreeOut(i) + 1)  ##DEFAULT infinite GAMMA
                    if __debug__:
                        for nei in self.__graph.iterNeighbors(i):
                            assert self.__colors[i] == self.__colors[nei]
                    break;
                assert (self.__graph.degreeOut(i) - min_freq) / self.__payoff[i] >= 1.0
                sum_gamma += (self.__graph.degreeOut(i) - min_freq) / self.__payoff[i]

        return sum_gamma / self.__graph.numberOfNodes()

    def computeLLLThreshold(self):
        bound = math.log(self.__maxoutdeg) + math.log(self.__maxindeg);

        print("LOG(maxoutdeg)+LOG(maxindeg):", bound)

        global_constant = self.__findConst();

        print("global_constant:", global_constant)

        if global_constant < 1:
            self.__not_lemma_valid = True
            print("NOT LEMMA VALID")

        for i in range(self.__graph.numberOfNodes()):
            if self.__graph.degreeOut(i) == 0:
                continue
            assert self.__graph.degreeOut(i) > 0

            if self.__graph.degreeOut(i) >= math.floor(global_constant * bound):  # numerical approximation
                continue;
            else:
                print(self.__graph.degreeOut(i), global_constant * bound)
                raise Exception('Unexpected global constant behaviour')

        first_term = len(self.__available) / (len(self.__available) - 1)

        numer = 3 * first_term * (bound + math.log(4) + math.log(
            len(self.__available)))  # JAIR
        max_global_term = 0

        for i in range(self.__graph.numberOfNodes()):
            if self.__graph.degreeOut(i) == 0:
                continue
            denom = ((1 / first_term) * self.__graph.degreeOut(i)) - 3 * (bound + math.log(4))
            contr = numer / denom
            max_global_term = max(max_global_term, contr)

        if self.__not_lemma_valid:
            return min(self.__maxoutdeg / len(self.__available), first_term + max_global_term)

        return first_term + max_global_term

    def __findConst(self):
        bound = math.log(self.__maxoutdeg) + math.log(self.__maxindeg)
        print("LOG(maxoutdeg)+LOG(maxindeg):", bound)
        costants = []

        for i in range(self.__graph.numberOfNodes()):
            if self.__graph.degreeOut(i) == 0:
                continue
            costants.append(self.__graph.degreeOut(i) / bound)

        return min(costants)

    def LLLColoring(self, MAX_ITERATIONS):

        self.__colors = [random.choice(self.__available) for i in range(self.__graph.numberOfNodes())]
        self.__computePayoffs()
        self.__LLL_THRESHOLD = self.computeLLLThreshold()

        print("LLL_T:", self.__LLL_THRESHOLD)
        unhappy_trend = []
        unhappy, num_unhappy = self.__getRandomGammaUnhappyVertex()
        unhappy_trend.append(num_unhappy / self.__graph.numberOfNodes())
        itrs = 0
        bar = IncrementalBar('LLL_Resampling:', max=MAX_ITERATIONS)
        gamma_value, is_NE, fraction_value_NE, is_gamma_NE, fraction_value_gamma_NE = self.nashStatus()

        while unhappy[0] != NULL_VERTEX and gamma_value > self.__LLL_THRESHOLD and itrs < MAX_ITERATIONS \
                and is_gamma_NE == False:
            self.__resampling(unhappy[0])
            self.__computePayoffs()
            bar.next()
            unhappy, num_unhappy = self.__getRandomGammaUnhappyVertex()
            unhappy_trend.append(num_unhappy / self.__graph.numberOfNodes())
            gamma_value, is_NE, fraction_value_NE, is_gamma_NE, fraction_value_gamma_NE = self.nashStatus()
            itrs += 1

        bar.finish()

        return itrs, unhappy_trend

    def __resampling(self, vertex):
        # JAIR
        vertices_to_recolor = set()
        for node in self.__graph.iterNeighbors(vertex):
            self.__colors[node] = random.choice(self.__available)
        # The events in which v’s neighbours are involved in their neighbours’ unhappiness
        #     for altronodo in self.__graph.iterInNeighbors(nodo):
        #         mfeq,colset = self.__isVertexGammaNEUnhappy(altronodo)
        #         if mfeq != NULL_VERTEX:
        #             #vertices_to_recolor.add(altronodo)
        #             vertices_to_recolor.add(nodo)
        #
        # # The events I_w, for any w != v, that make w unhappy, namely there exists a directed edge (v,w)
        # for nodo in self.__graph.iterNeighbors(vertex):
        #     mfeq,colset = self.__isVertexGammaNEUnhappy(nodo)
        #     if mfeq != NULL_VERTEX: #infelice
        #         vertices_to_recolor.add(nodo)
        #
        # # The events I_w, for any w != v where v contributes to w′s unhappiness, namely there exists a directed edge
        # # (w,v) (<= \delta(v_i) )
        # for nodo in self.__graph.iterInNeighbors(vertex):
        #     mfeq,colset = self.__isVertexGammaNEUnhappy(nodo)
        #     if mfeq != NULL_VERTEX: #infelice
        #         vertices_to_recolor.add(nodo)

        # for v in vertices_to_recolor:
        #     self.__colors[v] = random.choice(self.__available)

    def efficientResampling(self, vertex):
        # JAIR
        vertices_to_recolor = set()
        for node in self.__graph.iterNeighbors(vertex):
            self.__colors[node] = random.choice(self.__available)
            self.__recompute_status[node] = 1
            for neigh in self.__graph.iterInNeighbors(node):
                self.__recompute_status[neigh] = 1
                self.__updatePayoff(neigh)
            self.__updatePayoff(node)
        self.__recompute_status[vertex] = 1
        self.__updatePayoff(vertex)
        #     # The events in which v’s neighbours are involved in their neighbours’ unhappiness
        #     for altronodo in self.__graph.iterInNeighbors(nodo):
        #         mfeq, colset = self.__isVertexGammaNEUnhappy(altronodo)
        #         if mfeq != NULL_VERTEX:
        #             # vertices_to_recolor.add(altronodo)
        #             vertices_to_recolor.add(nodo)
        #
        #             # The events I_w, for any w != v, that make w unhappy, namely there exists a directed edge (v,w)
        # for nodo in self.__graph.iterNeighbors(vertex):
        #     mfeq, colset = self.__isVertexGammaNEUnhappy(nodo)
        #     if mfeq != NULL_VERTEX:  # infelice
        #         vertices_to_recolor.add(nodo)
        #
        # # The events I_w, for any w != v where v contributes to w′s unhappiness, namely there exists a directed edge
        # # (w,v) (<= \delta(v_i) )
        # for nodo in self.__graph.iterInNeighbors(vertex):
        #     mfeq, colset = self.__isVertexGammaNEUnhappy(nodo)
        #     if mfeq != NULL_VERTEX:  # infelice
        #         vertices_to_recolor.add(nodo)
        #
        # for v in vertices_to_recolor:
        #     self.__colors[v] = random.choice(self.__available)
        #     self.__recompute_status[v] = 1
        #     self.__updatePayoff(v)
        # for neig in self.__graph.iterInNeighbors(v):
        #     self.__recompute_status[neig] = 1
        #     self.__updatePayoff(neig)

        # self.__recompute_status[vertex] = 1
        # self.__updatePayoff(vertex)

    def Approx1(self):
        v = self.__getFirstNonColoredVertex();
        L = []
        bar = IncrementalBar('Coloring Vertices:', max=self.__graph.numberOfNodes())

        while v != NULL_VERTEX:
            L = [v];
            x = v;
            i = self.__getNonColoredNeighbor(x);
            while i != NULL_VERTEX:
                try:  # i in L
                    pos = L.index(i)
                except ValueError:
                    x = i;
                    L.append(i);
                    i = self.__getNonColoredNeighbor(x);
                else:
                    Lprime = L[pos:]
                    if len(Lprime) % 2 == 0:
                        self.__colorListByTwoColors(Lprime, [self.__available[0], self.__available[1]], bar);
                        x = i;
                        i = self.__getNonColoredNeighbor(x);
                        break;
                    else:

                        self.__colorListByThreeColors(Lprime, i,
                                                      [self.__available[0], self.__available[1], self.__available[2]],
                                                      bar);
                        x = i;
                        i = self.__getNonColoredNeighbor(x);
                        break;

            if v == x and self.__colors[v] == NULL_VERTEX:
                if self.__graph.degreeOut(v) > 0:
                    min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(v)
                    # payoff,colorset = self.__colorsDifferentThanCurrentThatInduceMaximumPayoff(v);
                    self.__colors[v] = random.choice(colorset);
                    bar.next()
                else:
                    self.__colors[v] = random.choice(self.__available)
                    bar.next()

            if v != x and self.__colors[x] == NULL_VERTEX:
                if self.__graph.degreeOut(x) > 0:
                    min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(x)

                    self.__colors[x] = random.choice(colorset);
                    bar.next()
                    suited_colors = []
                    for cl in self.__available:
                        if cl != self.__colors[x]:
                            suited_colors.append(cl);
                            if len(suited_colors) == 2:
                                break;
                    assert len(suited_colors) == 2;
                    self.__colorListByThreeColors(L, x, [suited_colors[0], suited_colors[1], self.__colors[x]], bar);

                else:
                    self.__colorListByTwoColors(L, [self.__available[0], self.__available[1]], bar);

            v = self.__getFirstNonColoredVertex();
        bar.finish()
        self.__computePayoffs();
        if __debug__:
            for i in range(self.__graph.numberOfNodes()):
                assert self.__payoff[i] >= 1 or self.__graph.degreeOut(i) == 0

    def __colorListByTwoColors(self, L, cols, barra):
        counter = 0;
        color_to_use = cols[counter];
        # color vertices in list with two colors, alternating

        for vertex in L:
            if self.__colors[vertex] == NULL_VERTEX:
                barra.next()
            self.__colors[vertex] = color_to_use;

            counter = (counter + 1) % 2;
            color_to_use = cols[counter];

    def __colorListByThreeColors(self, L, i, cols, barra):
        counter = 0;
        color_to_use = cols[counter];

        for vertex in L:
            # color vertices in list with two colors, alternating
            # color vertex i with a third different color
            if self.__colors[vertex] == NULL_VERTEX:
                barra.next()
            if i == vertex:
                self.__colors[vertex] = cols[2]
            else:
                self.__colors[vertex] = color_to_use;
                counter = (counter + 1) % 2;
                color_to_use = cols[counter];

    def Approx3(self, epsilon):

        kprime = math.ceil((3 * (1 + epsilon)) / epsilon)
        print("== KPRIME: " + str(kprime))
        # print("KPRIME: ",kprime)

        i = 1;
        bar = IncrementalBar('Coloring True Vertices:', max=self.__graph.numberOfNodes())
        # vprime is vertex set of graph_prime
        graph_prime = nk.graph.Graph(self.__graph.numberOfNodes(), weighted=False, directed=True)
        present_prime = [True for i in range(graph_prime.numberOfNodes())]
        # print(self.__available)
        while present_prime.count(True) > 0:

            # print("iteration: ",str(i))

            # color_indices=[((i-1)*kprime)+j for j in range(kprime) if ((i-1)*kprime)+j<len(self.__available)]

            usd = self.__listOfUsedColors();
            # print("CURRENTLY USED: ",usd)
            available_colors = [el for el in self.__available if el not in usd];

            if len(available_colors) > kprime:
                new_colors = [random.choice(available_colors) for i in range(kprime)];
            else:
                new_colors = available_colors;
            # print("USED INDICES:",color_indices)
            # print("REMAINING VERTICES:",len(Vprime))
            # print("ATTEMPTED COLORS:",new_colors)

            # indexOf = {}
            # for vtx in Vprime:
            #     indexOf[vtx] = Vprime.index(vtx)
            assert graph_prime.numberOfEdges() == 0;
            for u, v in self.__graph.iterEdges():
                if present_prime[u] == True:
                    if present_prime[v] == True:
                        graph_prime.addEdge(u, v);

            und_graph_prime = graphtools.toUndirected(graph_prime)
            # print("HERE 1")
            if und_graph_prime.isDirected() == True:
                raise Exception('DIRECTED GRAPH!')
            # print("HERE 2")

            colored_und_graph_prime = und.undColoredGraph(und_graph_prime, new_colors)

            colored_und_graph_prime.undirectedKColoring()
            for vertex in range(graph_prime.numberOfNodes()):
                if present_prime[vertex]:
                    if graph_prime.degreeOut(vertex) >= math.ceil(und_graph_prime.degree(vertex) / 3):
                        self.__colors[vertex] = colored_und_graph_prime.colors[vertex];
                        bar.next()
                        present_prime[vertex] = False;
            i += 1;
            graph_prime.removeAllEdges();
            bar.next()
        bar.finish()
        if self.numberOfUsedColors() > round((6 * (1 + epsilon)) / epsilon) * math.log2(self.__graph.numberOfNodes()):
            raise Exception('Too many colors used')
        self.__computePayoffs();
