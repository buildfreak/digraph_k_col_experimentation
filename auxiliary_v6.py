#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
import csv
import networkit as nk 
import math

NAME={}
NAME[0]='RND'
NAME[1]='AP1'
#NAME[2]='AP3'
NAME[3]='LLL'
NAME[4]='ADY'
NAME[5]='ALL'
NAME[6]='DYN'


CODE={}
CODE['RND']=0
CODE['AP1']=1
#CODE['AP3']=2
CODE['LLL']=3
CODE['ADY']=4
CODE['ALL']=5
CODE['DYN']=6

def writeedge(f,u,v,w,eid):
    assert u!=v;
    f.write(str(0)+" "+str(u)+" "+str(v)+" "+str(int(w))+"\n")
 
def hist2nk(name):
    fhandle = open(name, "r")
    print("Reading:",name)
    firstline = True
    for line in fhandle:
        # print(line)
        if firstline == True:
            fields = line.split(" ");
            firstline = False
            # print(fields)
            n = int(fields[0])
            m = int(fields[1])
            weighted = int(fields[2])==1
            directed = int(fields[3])==1
            graph = nk.graph.Graph(n,weighted,directed)
        else:
            fields = line.split(" ");
            # print(fields)
            graph.addEdge(int(fields[1]),int(fields[2]),int(fields[3]))
                

    assert graph.numberOfEdges()==m
    wgraph = nk.graph.Graph(graph.numberOfNodes(),graph.isWeighted(),graph.isDirected())
    assert graph.numberOfNodes()==wgraph.numberOfNodes()
    if weighted==True:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i,v,graph.weight(i,v))
    else:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i,v);
    wgraph.removeMultiEdges()
    wgraph.removeSelfLoops()
    return wgraph;

def ntwk2hist(name,graph):
    print("saving:",name)
    f = open(name, "w")
    if graph.isWeighted():
        we = 1;
    else:
        we = 0;
    if graph.isDirected():
        di = 1;
    else:
        di = 0;
            
    graph.removeMultiEdges()   
    graph.removeSelfLoops() 
    f.write(str(graph.numberOfNodes())+" "+str(graph.numberOfEdges())+" "+str(we)+" "+str(di)+"\n")
    for u,v in graph.iterEdges():
        writeedge(f,u,v,0,0)
    
    f.close()
    print("saved:",name)

def findConst(graph):
    maxoutdeg = max([graph.degreeOut(i) for i in range(graph.numberOfNodes())])
    maxindeg = max([graph.degreeIn(i) for i in range(graph.numberOfNodes())])
    bound = math.log(maxoutdeg)+math.log(maxindeg);
    costants = []
    
    for i in range(graph.numberOfNodes()): 
        if graph.degreeOut(i)==0:
            continue
        costants.append(graph.degreeOut(i)/bound)
        
    return min(costants)

def computeLLLThreshold(graph,num_k):
        
        
        maxoutdeg = max([graph.degreeOut(i) for i in range(graph.numberOfNodes())])
        maxindeg = max([graph.degreeIn(i) for i in range(graph.numberOfNodes())])
        bound = math.log(maxoutdeg)+math.log(maxindeg);
        
        global_constant = findConst(graph)
        

        for i in range(graph.numberOfNodes()):
            if graph.degreeOut(i)==0:
                continue
            assert graph.degreeOut(i)>0
        
            if graph.degreeOut(i)>=math.floor(global_constant*bound):#numerical approximation
                continue;
            else:
                raise Exception('Unexpected global constant behaviour')
                
        first_term =  (num_k)/((num_k)-1)
        
        numer = 3*first_term*(bound+math.log(4))
        max_global_term = 0
        
        for i in range(graph.numberOfNodes()):
            denum = ((1/first_term)*graph.degreeOut(i))-3*(bound+math.log(4))
            contr = numer/denum
            max_global_term = max(max_global_term,contr)
            
        return first_term+max_global_term    
        

def headermulti(statsfile):
    with open(statsfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_name", "vertices", "arcs","gamma_unhappy", "nash_unhappy", "avgdeg", "maxdeg", "k", "algo", "numused","avgpayoff","isGammaNash", "isNash", "gamma", "LLL-Threshold", "time","iterations","epsilon", "lemma-valid"])

def dumpOneShotData(gname, cgraph, elap, its, kvalue, algo, epsilon, gamma, isnashboolean, fractionNE, isgammanash, fractionGammaNE):
    
    maxdeg = max([cgraph.getGraph().degreeOut(i) for i in range(cgraph.getGraph().numberOfNodes())])
    avgdeg = sum([cgraph.getGraph().degreeOut(i) for i in range(cgraph.getGraph().numberOfNodes())])/cgraph.getGraph().numberOfNodes()
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
 
    with open(gname+"_"+str(algo)+"_"+date_time+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["graph_name", "vertices", "arcs","gamma_unhappy", "nash_unhappy", "avgdeg", "maxdeg", "k", "algo", "numused","avgpayoff","isGammaNash", "isNash", "gamma", "LLL-Threshold", "time","iterations","epsilon", "lemma-valid"])
        writer.writerow([gname, cgraph.getGraph().numberOfNodes(), cgraph.getGraph().numberOfEdges(),round(fractionGammaNE,4), round(fractionNE,4), avgdeg, maxdeg,kvalue,algo,cgraph.numberOfUsedColors(),round(sum(cgraph.getPayoffs())/len(cgraph.getPayoffs()),2),str(isgammanash), str(isnashboolean),round(gamma,4), round(cgraph.getLLLThreshold(),4),str(round(elap,6)),str(round(its,2)),epsilon, str(not cgraph.getNotLemmaValid())])
    
    with open(gname+"_"+str(algo)+"_"+date_time+'.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)
    
def dumpMultiData(csvfile, gname, cgraph,elap,its, kvalue,algo,epsilon, avgdeg, maxdeg, isnashboolean, isgammanash, gamma , fractionNE, fractionGammaNE):    
     
    writer = csv.writer(csvfile)
    writer.writerow([gname, cgraph.getGraph().numberOfNodes(), cgraph.getGraph().numberOfEdges(),round(fractionGammaNE,4), round(fractionNE,4), avgdeg, maxdeg,kvalue,algo,cgraph.numberOfUsedColors(),round(sum(cgraph.getPayoffs())/len(cgraph.getPayoffs()),2),str(isgammanash), str(isnashboolean),round(gamma,4), round(cgraph.getLLLThreshold(),4),str(round(elap,6)),str(round(its,2)),epsilon, str(not cgraph.getNotLemmaValid())])
    