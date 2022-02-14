#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkit
import math

from numpy import minimum
from colored_graph_v6 import coloredGraph as cg
from datetime import datetime
from timeit import default_timer as timer
import csv
import auxiliary_v6 as auxiliary
from progress.bar import IncrementalBar
import argparse

NAME={}
NAME[0]='RND'
NAME[1]='AP1'
#NAME[2]='AP3'
NAME[3]='LLL'
NAME[4]='ADY'
NAME[5]='ALL'
NAME[6]='DYN'

NULL_EPSILON = -1

CODE={}
CODE['RND']=0
CODE['AP1']=1
#CODE['AP3']=2
CODE['LLL']=3
CODE['ADY']=4
CODE['ALL']=5
CODE['DYN']=6



if __name__ == "__main__":
    
    helpstring = "";
    
    for k,v in NAME.items():
        helpstring += "#"+str(CODE[v])+" for "+v+"\n";


    NOISE_ITERS = 3;    
    parser = argparse.ArgumentParser(description='KColoring Nash',add_help=True)  
    parser.add_argument("--g",metavar="GRAPH_NAME", required=True, default="", help="path to input graph")
    parser.add_argument("--a",metavar="ALGO_TO_TEST", required=True, default=0,help=helpstring)
    parser.add_argument("--k",metavar="NUM_KOLORS", required=False, default=3,help="Number of colors to use, default=3; -1 if multi-k")
    parser.add_argument("--i",metavar="NUM_DYN_ITERS", required=False, default=1, help="Number of iterations to use for dynamics estimation, default=1")
    
    
    args = parser.parse_args()
    
    if int(args.a) not in  range(8):
        parser.print_help();
        raise Exception('ALGO_TO_TEST out of range')
        
    IN_KOLORS = int(args.k)
    ITERATIONS = int(args.i);

    if IN_KOLORS < 3 and IN_KOLORS != -1:
        parser.print_help();
        raise Exception('KOLORS < 3')
        
    if int(args.a)!=CODE['DYN']: 
        print("WARNING - GIVEN ITERATIONS IGNORED")
        ITERATIONS=1
    
    if ITERATIONS < 1:
        parser.print_help();
        raise Exception('ITERATIONS < 1')

        
    print("READING GRAPH: "+str(args.g))
    graph_name = str(args.g)
    graph = auxiliary.hist2nk(graph_name)
    mindeg = min([graph.degreeOut(i) for i in range(graph.numberOfNodes()) if graph.degreeOut(i) > 3])
    maxdeg = max([graph.degreeOut(i) for i in range(graph.numberOfNodes())])
    minimum = min(mindeg, 6)
    if  graph.isDirected()==False:
        raise Exception('UNDIRECTED GRAPH!')
    
    if IN_KOLORS == -1:
        IN_KOLORS = [k for k in range(minimum, maxdeg+1, int((maxdeg - minimum)//5))]
    else:
        IN_KOLORS = [IN_KOLORS]
    for KOLORS in IN_KOLORS:
        print("KOLORS: "+str(KOLORS))
        # graph features 
        print("VERTICES: "+str(graph.numberOfNodes()), "ARCS: "+str(graph.numberOfEdges()))
        print("DIRECTED: "+str(graph.isDirected()))
        print("WEIGHTED: "+str(graph.isWeighted()))
        print("EXPERIMENT TYPE:",str(NAME[int(args.a)]))
        
        col_graph = cg(g=graph, kolors=[j for j in range(KOLORS)])

        
        print("STARTING # COLORS:",str(len(col_graph.getAvailable())))
        UPPERBOUND_ITERATIONS = math.floor(graph.numberOfNodes()*math.log2(graph.numberOfNodes()))

        if int(args.a)==CODE['RND']: #random coloring
        
            print("== RANDOM COLORING ==")    
            if NAME[int(args.a)]!=NAME[CODE['RND']]:
                raise Exception('wrong algo name')
            for _ in range(NOISE_ITERS):
                cpu=timer();
                col_graph.randomColoring();
                elapsed=timer()-cpu
                col_graph.setLLLThreshold(col_graph.computeLLLThreshold())
                gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                if col_graph.colored()==False:
                    raise Exception('Non fully colored graph')
                auxiliary.dumpOneShotData(graph_name,col_graph,elapsed,ITERATIONS,str(KOLORS),NAME[int(args.a)],NULL_EPSILON, gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash)
                if col_graph.numberOfUsedColors() > KOLORS:
                    raise Exception('Anomalous usage of KOLORS')
                col_graph.reset()

        elif int(args.a)==CODE['AP1']: #Approx1 Coloring 
        
            print("== APPROX1 COLORING ==")
            if NAME[int(args.a)]!=NAME[CODE['AP1']]:
                raise Exception('wrong algo name')
            cpu=timer();
            col_graph.Approx1();
            elapsed=timer()-cpu
            col_graph.setLLLThreshold(col_graph.computeLLLThreshold())
            gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
            if col_graph.colored()==False:
                raise Exception('Non fully colored graph')
            auxiliary.dumpOneShotData(graph_name,col_graph,elapsed,ITERATIONS,str(KOLORS),NAME[int(args.a)],NULL_EPSILON, gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash)
            
            if col_graph.numberOfUsedColors() > KOLORS:
                raise Exception('Anomalous usage of KOLORS')
                

        elif int(args.a)==CODE['ADY']:
            
            if NAME[int(args.a)]!=NAME[CODE['ADY']]:
                raise Exception('wrong algo name')
            print("== DYN_ALGO COLORING ==")    
            
            cpu=timer();
            bar = IncrementalBar('Iterations:', max = UPPERBOUND_ITERATIONS)
            
            col_graph.randomColoring();
            col_graph.setLLLThreshold(col_graph.computeLLLThreshold())

            gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
            
            itrs = 1;
            bar.next()

            while itrs < UPPERBOUND_ITERATIONS and isnashboolean==False:
                #evolve
                col_graph.improve()
                gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                bar.next()
                itrs+=1;    

            elapsed=timer()-cpu 

            bar.finish()
            if col_graph.colored()==False:
                raise Exception('Non fully colored graph')
            if col_graph.numberOfUsedColors() > KOLORS:
                raise Exception('Anomalous usage of KOLORS')
                
            auxiliary.dumpOneShotData(graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[int(args.a)],NULL_EPSILON, gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash)

        
        elif int(args.a)==CODE['ALL']:
            avgdeg = sum([col_graph.getGraph().degreeOut(i) for i in range(col_graph.getGraph().numberOfNodes())])/col_graph.getGraph().numberOfNodes()
            maxdeg = max([col_graph.getGraph().degreeOut(i) for i in range(col_graph.getGraph().numberOfNodes())])

            now = datetime.now() # current date and time
            date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
            statsfile = graph_name+"_"+NAME[CODE['ALL']]+"_"+date_time+'.csv';
            auxiliary.headermulti(statsfile)
            cpu=timer();
            with open(statsfile, 'a', newline='') as csvfile:
                 
                print("== RANDOM COLORING ==")  
                for iteration in range(NOISE_ITERS):    #to reduce noise due to randomness 
                    col_graph.reset()
                    col_graph.setLLLThreshold(col_graph.computeLLLThreshold())
                    cpu=timer();
                    col_graph.randomColoring();
                    elapsed=timer()-cpu 
                    if col_graph.colored()==False:
                        raise Exception('Non fully colored graph')
                    if col_graph.numberOfUsedColors() > KOLORS:
                        raise Exception('Anomalous usage of KOLORS')
                    gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                    auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,ITERATIONS,str(KOLORS),NAME[CODE['RND']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)
                    
                print("== APPROX1 COLORING ==")    
                col_graph.reset()
                col_graph.setLLLThreshold(col_graph.computeLLLThreshold())
                cpu=timer();
                col_graph.Approx1()
                elapsed=timer()-cpu 
                if col_graph.colored()==False:
                    raise Exception('Non fully colored graph')
                if col_graph.numberOfUsedColors() > KOLORS:
                    raise Exception('Anomalous usage of KOLORS')
                gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,ITERATIONS,str(KOLORS),NAME[CODE['AP1']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)
            
                print("== LLL RANDOM COLORING ==")  
                for iteration in range(NOISE_ITERS):    #to reduce noise due to randomness 
                    col_graph.reset()
                    cpu=timer();
                    
                    itrs = 0
                    itrs, unhappy_trend = col_graph.LLLColoring(UPPERBOUND_ITERATIONS);
                    elapsed=timer()-cpu 

                    if col_graph.colored()==False:
                        raise Exception('Non fully colored graph')
                    if col_graph.numberOfUsedColors() > KOLORS:
                        raise Exception('Anomalous usage of KOLORS')
                    if len(unhappy_trend) > 1:
                        auxiliary.plotUnhappyTrend(graph_name, iteration, unhappy_trend)
                    gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                    auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[CODE['LLL']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)

                print("== DYN_ALGO COLORING ==")  
                for iteration in range(NOISE_ITERS):
                    col_graph.reset()
                    col_graph.setLLLThreshold(col_graph.computeLLLThreshold())

                    cpu=timer();
                    
                    bar = IncrementalBar('Iterations:', max = UPPERBOUND_ITERATIONS)
        
                    col_graph.randomColoring()
                    gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                    
        
                    itrs = 1;
                    bar.next()
                    while itrs < UPPERBOUND_ITERATIONS and isnashboolean==False:
                        #evolve
                        col_graph.improve()
                        gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                        bar.next()
                        itrs+=1;    
                        
                    elapsed=timer()-cpu 
        
                    bar.finish()
                    if col_graph.colored()==False:
                        raise Exception('Non fully colored graph')
                    if col_graph.numberOfUsedColors() > KOLORS:
                        raise Exception('Anomalous usage of KOLORS')
                    auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[CODE['ADY']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)
        
                
            with open(statsfile, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    print(row)
                
        elif int(args.a)==CODE['DYN']:
            bar = IncrementalBar('Iterations:', max = UPPERBOUND_ITERATIONS)


            avgdeg = sum([col_graph.getGraph().degreeOut(i) for i in range(col_graph.getGraph().numberOfNodes())])/col_graph.getGraph().numberOfNodes()
            maxdeg = max([col_graph.getGraph().degreeOut(i) for i in range(col_graph.getGraph().numberOfNodes())])

            print("== DYNAMICS ==")    
            print("== ITERATIONS ==",graph.numberOfNodes())    
            now = datetime.now() # current date and time
            date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
            if NAME[int(args.a)]!=NAME[CODE['DYN']]:
                raise Exception('wrong algo name')
            statsfile = graph_name+"_"+NAME[CODE['DYN']]+"_"+str(KOLORS)+"_"+date_time+'.csv';
            auxiliary.headermulti(statsfile)

        
            cpu=timer();
            with open(statsfile, 'a', newline='') as csvfile:
                #start by random coloring
                col_graph.randomColoring()
                col_graph.setLLLThreshold(col_graph.computeLLLThreshold())
                gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                itrs = 0;
                elapsed=timer()-cpu 
                auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[CODE['DYN']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)
                itrs+=1;
                bar.next()
                while itrs < UPPERBOUND_ITERATIONS and isnashboolean==False:
                    #evolve
                    col_graph.improve()
                    gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
                    
                    elapsed=timer()-cpu 
                    auxiliary.dumpMultiData(csvfile,graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[CODE['DYN']],NULL_EPSILON,avgdeg,maxdeg,isnashboolean, isgammanashboolean, gammavalue,fractionvalue_nash, fractionvalue_gamma_nash)
                    itrs+=1;
                    bar.next()

                
                bar.finish()

                if col_graph.colored()==False:
                    raise Exception('Non fully colored graph')
                if col_graph.numberOfUsedColors() > KOLORS:
                    raise Exception('Anomalous usage of KOLORS')
                print("== IS NASH ==",isnashboolean)    
                print("== GAMMA ==",gammavalue) 
                
            
        else:    
            assert int(args.a)==CODE['LLL']      
            if NAME[int(args.a)]!=NAME[CODE['LLL']]:
                raise Exception('wrong algo name')
            print("== LLL COLORING ==")    
            cpu=timer();
            itrs = col_graph.LLLColoring(UPPERBOUND_ITERATIONS)[0];
            elapsed=timer()-cpu 
            if col_graph.colored()==False:
                raise Exception('Non fully colored graph')
            gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash = col_graph.nashStatus()
            auxiliary.dumpOneShotData(graph_name,col_graph,elapsed,itrs,str(KOLORS),NAME[int(args.a)],NULL_EPSILON, gammavalue, isnashboolean, fractionvalue_nash, isgammanashboolean, fractionvalue_gamma_nash)

