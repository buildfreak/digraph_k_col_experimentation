#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:35:46 2020

@author: anonym
"""
import sys
import random
DEFAULT_NULL_COLOR = -1
DEFAULT_PAYOFF = 1
NULL_VERTEX = -1
INFTY=sys.maxsize;

class undColoredGraph:
    
    def __init__(self, g, kolors):
        # -1 DEFAULT_NULL_COLOR
        self.graph = g
        self.colors = [DEFAULT_NULL_COLOR for i in range(self.graph.numberOfNodes())]
        if __debug__:
            self.__payoff = [DEFAULT_PAYOFF for i in range(self.graph.numberOfNodes())]
        self.__available = kolors
        # print(kolors)
    # def __reset(self):
    #     self.colors = [DEFAULT_NULL_COLOR for i in range(self.graph.numberOfNodes())]
    #     self.__payoff = [DEFAULT_PAYOFF for i in range(self.graph.numberOfNodes())]
                        
    def __existsNonColoredVertex(self):
        for i in range(self.graph.numberOfNodes()):
            if self.colors[i]==DEFAULT_NULL_COLOR:
                return i;
        return NULL_VERTEX;
    
    def undirectedKColoring(self):
    
        #start from arbitraty coloring
        self.colors = [random.choice(self.__available) for i in range(self.graph.numberOfNodes())];
        
        while True:
            unhappyvertex, mf, colorset = self.__existsUnhappyVertex();  
                        
            # print(unhappyvertex)
            if unhappyvertex == NULL_VERTEX:
                break;

            # print("unhappy vertex: " +str(unhappyvertex))
            # candidate_color = self.__minColorAmongNeighbors(unhappyvertex);
            # print("color of vertex: " +str(self.colors[unhappyvertex]))
            # print("candidate color: " +str(candidate_color))
            assert self.colors[unhappyvertex] not in colorset;
            self.colors[unhappyvertex] = random.choice(colorset);
            # print("colored vertex: " +str(unhappyvertex)+" of "+str(candidate_color))
            
        
        assert self.__existsNonColoredVertex()==NULL_VERTEX
        if __debug__:
            self.__computePayoff();
            # print(self.colors)
            # print(self.__payoff)
            assert self.__nonImprovablePayoff()==True;          
            assert self.isNash()
        
    def __computePayoff(self):
        self.__payoff = [DEFAULT_PAYOFF for i in range(self.graph.numberOfNodes())]
        for i in range(self.graph.numberOfNodes()):
            for n in self.graph.iterNeighbors(i):
                if self.colors[i] != self.colors[n]:
                    self.__payoff[i]+=1;    
                    
    def __colorsWithMinimumFrequencyInNeighborhood(self,vertex):
        
        base_frequencies = {}
        for i in self.__available:
            base_frequencies[i]=0;
            
        for n in self.graph.iterNeighbors(vertex):
            if self.colors[n]!=DEFAULT_NULL_COLOR:
                base_frequencies[self.colors[n]]+=1;
                    
        min_f_color = min(base_frequencies.keys(), key=(lambda k: base_frequencies[k]))         
        min_frequency_colors=[]
        for k in base_frequencies.keys():
            if base_frequencies[k]==base_frequencies[min_f_color]:
                min_frequency_colors.append(k)
        
        return base_frequencies[min_f_color],min_frequency_colors;   
            
    # def __minColorAmongNeighbors(self,vertex):
    #     color_frequencies = {}

    #     for c in self.__available:
    #         color_frequencies[c]=0;
        
    #     for n in self.graph.iterNeighbors(vertex):
    #         if self.colors[n]!=DEFAULT_NULL_COLOR:
    #             color_frequencies[self.colors[n]]+=1;
        
    #     min_frequency = INFTY
    #     min_frequency_color = DEFAULT_NULL_COLOR
    #     for key,value in color_frequencies.items():
    #         if value < min_frequency:
    #             min_frequency = value;
    #             min_frequency_color = key;         
    #     assert min_frequency_color == min(color_frequencies.keys(), key=(lambda k: color_frequencies[k]))
    #     assert min_frequency_color>DEFAULT_NULL_COLOR;

        
    #     return min_frequency_color;
   
    def __nonImprovablePayoff(self):
        for i in range(self.graph.numberOfNodes()):
            
            current_payoff = self.__payoff[i];
            current_color = self.colors[i];
           
            for alt_color in self.__available:
                if alt_color != current_color:
                    temp_payoff = 0;
                    for n in self.graph.iterNeighbors(i):
                        if alt_color != self.colors[n]:
                            temp_payoff+=1;
                    if temp_payoff > current_payoff:
                        return False;
        return True; 
    
    def isNash(self): 
        i,j,k = self.__existsUnhappyVertex()
        return i == NULL_VERTEX;
   
    def __existsUnhappyVertex(self):
        #a vertex v is unhappy if it has more neighbors of its color than of someother color
        for i in range(self.graph.numberOfNodes()):
            assert self.colors[i]!=DEFAULT_NULL_COLOR
            min_freq, colorset = self.__colorsWithMinimumFrequencyInNeighborhood(i)

            if self.colors[i] not in colorset:
                return i, min_freq, colorset; #unhappy found 

        return NULL_VERTEX, NULL_VERTEX, []; #no unhappy in the graph     