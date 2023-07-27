#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:46:35 2020

@author: anonym
"""
from datetime import datetime
import csv
from turtle import color
import networkit as nk
import pandas as pd
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import argparse
import shutil
import operator
import random
import matplotlib.ticker as ticker
import math
import statistics
import networkx as nx

NAMES = {}
NAMES["dir_erdos_1000_0.0125"] = '\\textsc{era}'
NAMES["dir_erdos_1000_0.025"] = '\\textsc{erb}'
NAMES["dir_erdos_1000_0.05"] = '\\textsc{erc}'
NAMES["dir_erdos_1000_0.1"] = '\\textsc{erd}'
NAMES["dir_erdos_1000_0.2"] = '\\textsc{ere}'

NAMES["moreno_health"] = '\\textsc{hea}'
NAMES["moreno_blogs"] = '\\textsc{blg}'
NAMES["google"] = '\\textsc{goo}'
NAMES["lux_scc1.bgr"] = '\\textsc{lux}'
NAMES["twitter"] = '\\textsc{twi}'

NAMES["weighted_p2p-Gnutella31.txt"] = '\\textsc{p2p}'
NAMES["oriented_wiki-Vote"] = '\\textsc{wvt}'
NAMES["out-maayan-faa"] = '\\textsc{flt}'
NAMES["oriented_oregon1_010331"] = '\\textsc{ore}'
NAMES["oriented_email-Eu-core"] = '\\textsc{ema}'

# NAMES["SCALE_0.25_web-Google.txt"]='\\textsc{gow}'
# NAMES["soc-sign-epinions.txt"]='\\textsc{epi}'
NAMES["oriented_p2p-Gnutella04"] = '\\textsc{spp}'
NAMES["linux.txt"] = '\\textsc{lin}'
NAMES["oriented_ca-GrQc"] = '\\textsc{rel}'

# NAMES["SCALE_0.25_sdir_FF2M.txt"]='\\textsc{for}'
NAMES["SCALE_0.2_out-amazon0601"] = '\\textsc{ama}'
# NAMES["SCALE_0.2_out.as-skitter"]='\\textsc{ski}'
# NAMES["SCALE_0.5_barabasi1M.bgr"]='\\textsc{bar}'
NAMES["SCALE_0.1_oriented_facebook"] = '\\textsc{fac}'

# new regular graphs
NAMES["rand_reg_3_10000"] = '\\textsc{rr3}'
NAMES["rand_reg_4_10000"] = '\\textsc{rr4}'
NAMES["rand_reg_5_10000"] = '\\textsc{rr5}'
NAMES["paley_601"] = '\\textsc{pl1}'
NAMES["paley_1181"] = '\\textsc{pl2}'

# new graphs
NAMES['academia'] = '\\textsc{aca}'
NAMES['arxiv'] = '\\textsc{arx}'
NAMES['polblogs'] = '\\textsc{pol}'
NAMES['us_air_traffic'] = '\\textsc{uat}'
NAMES['soc-Slashdot0902'] = '\\textsc{sld}'
NAMES['chicago'] = '\\textsc{chi}'
NAMES['dblp_cite'] = '\\textsc{dbl}'
NAMES['dbpedia_all'] = '\\textsc{dbp}'
NAMES['google_plus'] = '\\textsc{gop}'
NAMES['soc-academia'] = '\\textsc{sca}'

SHORT_NAMES = {}
SHORT_NAMES["dir_erdos_1000_0.0125"] = '\\erdosa'
SHORT_NAMES["dir_erdos_1000_0.025"] = '\\erdosb'
SHORT_NAMES["dir_erdos_1000_0.05"] = '\\erdosc'
SHORT_NAMES["dir_erdos_1000_0.1"] = '\\erdosd'
SHORT_NAMES["dir_erdos_1000_0.2"] = '\\erdose'

SHORT_NAMES["moreno_health"] = '\\health'
SHORT_NAMES["moreno_blogs"] = '\\blog'
SHORT_NAMES["google"] = '\\google'
SHORT_NAMES["lux_scc1.bgr"] = '\\lux'
SHORT_NAMES["twitter"] = '\\twitter'

SHORT_NAMES["weighted_p2p-Gnutella31.txt"] = '\\peer'
SHORT_NAMES["oriented_wiki-Vote"] = '\\wikiv'
SHORT_NAMES["out-maayan-faa"] = '\\flight'
SHORT_NAMES["oriented_oregon1_010331"] = '\\oregon'
SHORT_NAMES["oriented_email-Eu-core"] = '\\emaildata'

# SHORT_NAMES["SCALE_0.25_web-Google.txt"]='\\lgoogle'
# SHORT_NAMES["soc-sign-epinions.txt"]='\\epi'
SHORT_NAMES["oriented_p2p-Gnutella04"] = '\\speer'
SHORT_NAMES["linux.txt"] = '\\linux'
SHORT_NAMES["oriented_ca-GrQc"] = '\\collab'

# SHORT_NAMES["SCALE_0.25_sdir_FF2M.txt"]='\\forest'
SHORT_NAMES["SCALE_0.2_out-amazon0601"] = '\\amazon'
# SHORT_NAMES["SCALE_0.2_out.as-skitter"]='\\skitter'
# SHORT_NAMES["SCALE_0.5_barabasi1M.bgr"]='\\baraba'
SHORT_NAMES["SCALE_0.1_oriented_facebook"] = '\\facebook'

# new regular graphs
SHORT_NAMES["rand_reg_3_10000"] = '\\regrandthree'
SHORT_NAMES["rand_reg_4_10000"] = '\\regrandfour'
SHORT_NAMES["rand_reg_5_10000"] = '\\regrandfive'
SHORT_NAMES["paley_601"] = '\\regpaleysmall'
SHORT_NAMES["paley_1181"] = '\\regpaleylarge'

# new graphs
SHORT_NAMES['academia'] = '\\academia'
SHORT_NAMES['arxiv'] = '\\arxiv'
SHORT_NAMES['polblogs'] = '\\politics'
SHORT_NAMES['us_air_traffic'] = '\\usair'
SHORT_NAMES['soc-Slashdot0902'] = '\\slashdot'
SHORT_NAMES['chicago'] = '\\chicago'
SHORT_NAMES['dblp_cite'] = '\\dblp'
SHORT_NAMES['dbpedia_all'] = '\\dbpedia'
SHORT_NAMES['google_plus'] = '\\googleplus'
SHORT_NAMES['soc-academia'] = '\\socialacademia'

TYPES = {}
TYPES["dir_erdos_1000_0.0125"] = '\\textsc{random}'
TYPES["dir_erdos_1000_0.025"] = '\\textsc{random}'
TYPES["dir_erdos_1000_0.05"] = '\\textsc{random}'
TYPES["dir_erdos_1000_0.1"] = '\\textsc{random}'
TYPES["dir_erdos_1000_0.2"] = '\\textsc{random}'

TYPES["moreno_health"] = '\\textsc{human social}'
TYPES["moreno_blogs"] = '\\textsc{interaction}'
TYPES["google"] = '\\textsc{hyperlinks (local)}'
TYPES["lux_scc1.bgr"] = '\\textsc{road}'
TYPES["twitter"] = '\\textsc{digital social}'

TYPES["weighted_p2p-Gnutella31.txt"] = '\\textsc{internet}'
TYPES["oriented_wiki-Vote"] = '\\textsc{voting}'
TYPES["out-maayan-faa"] = '\\textsc{infrastructure}'
TYPES["oriented_oregon1_010331"] = '\\textsc{autonomous system}'
TYPES["oriented_email-Eu-core"] = '\\textsc{interaction}'

# TYPES["SCALE_0.25_web-Google.txt"]='\\textsc{hyperlinks (global)}'
# TYPES["soc-sign-epinions.txt"]='\\textsc{digital social}'
TYPES["oriented_p2p-Gnutella04"] = '\\textsc{internet}'
TYPES["linux.txt"] = '\\textsc{community}'
TYPES["oriented_ca-GrQc"] = '\\textsc{collaboration}'

# TYPES["SCALE_0.25_sdir_FF2M.txt"]='\\textsc{hybrid-random}'
TYPES["SCALE_0.2_out-amazon0601"] = '\\textsc{ratings}'
# TYPES["SCALE_0.2_out.as-skitter"]='\\textsc{communication}'
# TYPES["SCALE_0.5_barabasi1M.bgr"]='\\textsc{hybrid-random}'
TYPES["SCALE_0.1_oriented_facebook"] = '\\textsc{digital social}'

TYPES["rand_reg_3_10000"] = '\\textsc{random}'
TYPES["rand_reg_4_10000"] = '\\textsc{random}'
TYPES["rand_reg_5_10000"] = '\\textsc{random}'
TYPES["paley_601"] = '\\textsc{random}'
TYPES["paley_1181"] = '\\textsc{random}'

TYPES['academia'] = '\\textsc{human social}'
TYPES['arxiv'] = '\\textsc{digital social}'
TYPES['polblogs'] = '\\textsc{voting}'
TYPES['us_air_traffic'] = '\\textsc{traffic}'
TYPES['soc-Slashdot0902'] = '\\textsc{social}'
TYPES['chicago'] = '\\textsc{road}'
TYPES['dblp_cite'] = '\\textsc{citation}'
TYPES['dbpedia_all'] = '\\textsc{web}'
TYPES['google_plus'] = '\\textsc{web}'
TYPES['soc-academia'] = '\\textsc{social}'

NAME = {}
NAME[0] = 'RND'
NAME[1] = 'AP1'
NAME[2] = 'AP3'
NAME[3] = 'LLL'
NAME[4] = 'ADY'
NAME[5] = 'ALL'
NAME[6] = 'DYN'
NAME[7] = 'AV3'

CODE = {}
CODE['RND'] = 0
CODE['AP1'] = 1
CODE['AP3'] = 2
CODE['LLL'] = 3
CODE['ADY'] = 4
CODE['ALL'] = 5
CODE['DYN'] = 6
CODE['AV3'] = 7


def writeedge(f, u, v, w, eid):
    assert u != v;
    f.write(str(0) + " " + str(u) + " " + str(v) + " " + str(int(w)) + "\n")


def nde2nk(name):
    fhandle = open(name, "r")
    print("Reading:", name)
    firstline = True
    for line in fhandle:
        # print(line)
        if firstline == True:
            fields = line.split(" ")
            firstline = False
            # print(fields)
            n = int(fields[0])
            m = int(fields[1])
            graph = nk.graph.Graph(n, 0, 1)
        else:
            fields = line.split(" ")
            # print(fields)
            graph.addEdge(int(fields[0]), int(fields[1]), addMissing=True)

    # assert graph.numberOfEdges() == m
    wgraph = nk.graph.Graph(graph.numberOfNodes(), graph.isWeighted(), graph.isDirected())
    assert graph.numberOfNodes() == wgraph.numberOfNodes()
    for i in range(graph.numberOfNodes()):
        for v in graph.iterNeighbors(i):
            wgraph.addEdge(i, v)
    wgraph.removeMultiEdges()
    wgraph.removeSelfLoops()
    return wgraph


def hist2nk(name):
    fhandle = open(name, "r")
    print("Reading:", name)
    firstline = True
    for line in fhandle:
        # print(line)
        if firstline == True:
            fields = line.split(" ");
            firstline = False
            # print(fields)
            n = int(fields[0])
            m = int(fields[1])
            weighted = int(fields[2]) == 1
            directed = int(fields[3]) == 1
            graph = nk.graph.Graph(n, weighted, directed)
        else:
            fields = line.split(" ");
            # print(fields)
            graph.addEdge(int(fields[1]), int(fields[2]), int(fields[3]), addMissing=True)

    assert graph.numberOfEdges() == m
    wgraph = nk.graph.Graph(graph.numberOfNodes(), graph.isWeighted(), graph.isDirected())
    assert graph.numberOfNodes() == wgraph.numberOfNodes()
    if weighted == True:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i, v, graph.weight(i, v))
    else:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i, v);
    wgraph.removeMultiEdges()
    wgraph.removeSelfLoops()
    return wgraph;


def ntwk2hist(name, graph):
    print("saving:", name)
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
    f.write(str(graph.numberOfNodes()) + " " + str(graph.numberOfEdges()) + " " + str(we) + " " + str(di) + "\n")
    for u, v in graph.iterEdges():
        writeedge(f, u, v, 0, 0)

    f.close()
    print("saved:", name)


def findConst(graph):
    maxoutdeg = max([graph.degreeOut(i) for i in range(graph.numberOfNodes())])
    maxindeg = max([graph.degreeIn(i) for i in range(graph.numberOfNodes())])
    bound = math.log(maxoutdeg) + math.log(maxindeg);
    # print("LOG(maxoutdeg)+LOG(maxindeg):",bound)
    costants = []

    for i in range(graph.numberOfNodes()):
        if graph.degreeOut(i) == 0:
            # print("GLOBAL CONSTANT CANNOT BE FOUND, GAMMA NOT CONSTANT")
            # print("STOPPING CRITERION IS ITERATIONS OR NASH")
            continue
        costants.append(graph.degreeOut(i) / bound)

    return min(costants)


def computeLLLThreshold(graph, num_k):
    maxoutdeg = max([graph.degreeOut(i) for i in range(graph.numberOfNodes())])
    maxindeg = max([graph.degreeIn(i) for i in range(graph.numberOfNodes())])
    bound = math.log(maxoutdeg) + math.log(maxindeg);
    # print("LOG(maxoutdeg)+LOG(maxindeg):",bound)

    global_constant = findConst(graph)

    # print("global_constant:",global_constant)

    for i in range(graph.numberOfNodes()):
        if graph.degreeOut(i) == 0:
            continue
        assert graph.degreeOut(i) > 0

        if graph.degreeOut(i) >= math.floor(global_constant * bound):  # numerical approximation
            continue;
        else:
            # print(graph.degreeOut(i),global_constant*bound)
            raise Exception('Unexpected global constant behaviour')
            # time.sleep(5)

    first_term = (num_k) / ((num_k) - 1)

    numer = 3 * first_term * (bound + math.log(4))
    max_global_term = 0

    for i in range(graph.numberOfNodes()):
        denum = ((1 / first_term) * graph.degreeOut(i)) - 3 * (bound + math.log(4))
        contr = numer / denum
        max_global_term = max(max_global_term, contr)

    return first_term + max_global_term


# with open('your_file.txt', 'w') as f:
#     for item in my_list:
#         f.write("%s\n" % item)
def headermulti(statsfile):
    with open(statsfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["graph_name", "vertices", "arcs", "gamma_unhappy", "nash_unhappy", "avgdeg", "maxdeg", "k", "algo",
             "numused", "avgpayoff", "isGammaNash", "isNash", "gamma", "LLL-Threshold", "time", "iterations", "epsilon",
             "lemma-valid"])


def dumpOneShotData(gname, cgraph, elap, its, kvalue, algo, epsilon, gamma, isnashboolean, fractionNE, isgammanash,
                    fractionGammaNE):
    maxdeg = max([cgraph.getGraph().degreeOut(i) for i in range(cgraph.getGraph().numberOfNodes())])
    avgdeg = sum([cgraph.getGraph().degreeOut(i) for i in
                  range(cgraph.getGraph().numberOfNodes())]) / cgraph.getGraph().numberOfNodes()

    # fraction = cgraph.fractionOfUnhappyVertices()
    # isnashboolean = cgraph.isNash()
    # if gamma!=1. and isnashboolean==True:
    #     raise Exception('mismatch gamma vs nash')
    # if gamma==1. and isnashboolean==False:
    #     raise Exception('mismatch gamma vs nash')
    # if fraction!=0. and isnashboolean==True:
    #     print("Fraction:",fraction)
    #     raise Exception('mismatch unhappy vs nash')
    # if fraction==0. and isnashboolean==False:
    #     raise Exception('mismatch unhappy vs nash')

    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
    # check = '' cambia
    # if gamma > 100000:
    #     return
    with open(gname + "_" + str(algo) + "_" + date_time + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["graph_name", "vertices", "arcs", "gamma_unhappy", "nash_unhappy", "avgdeg", "maxdeg", "k", "algo",
             "numused", "avgpayoff", "isGammaNash", "isNash", "gamma", "LLL-Threshold", "time", "iterations", "epsilon",
             "lemma-valid"])
        writer.writerow(
            [gname, cgraph.getGraph().numberOfNodes(), cgraph.getGraph().numberOfEdges(), round(fractionGammaNE, 4),
             round(fractionNE, 4), avgdeg, maxdeg, kvalue, algo, cgraph.numberOfUsedColors(),
             round(sum(cgraph.getPayoffs()) / len(cgraph.getPayoffs()), 2), str(isgammanash), str(isnashboolean),
             round(gamma, 4), round(cgraph.getLLLThreshold(), 4), str(round(elap, 6)), str(round(its, 2)), epsilon,
             str(not cgraph.getNotLemmaValid())])

    with open(gname + "_" + str(algo) + "_" + date_time + '.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)


def dumpMultiData(csvfile, gname, cgraph, elap, its, kvalue, algo, epsilon, avgdeg, maxdeg, isnashboolean, isgammanash,
                  gamma, fractionNE, fractionGammaNE):
    # if gamma!=1. and isnashboolean==True:
    #     raise Exception('mismatch gamma vs nash')
    # if gamma==1. and isnashboolean==False:
    #     raise Exception('mismatch gamma vs nash')
    # if fractionNE!=0. and isnashboolean==True:
    #     print("Fraction:",fractionNE)
    #     raise Exception('mismatch unhappy vs nash')
    # if fractionNE==0. and isnashboolean==False:
    #     raise Exception('mismatch unhappy vs nash')

    # if gamma > 10000:
    #     return
    writer = csv.writer(csvfile)
    writer.writerow(
        [gname, cgraph.getGraph().numberOfNodes(), cgraph.getGraph().numberOfEdges(), round(fractionGammaNE, 4),
         round(fractionNE, 4), avgdeg, maxdeg, kvalue, algo, cgraph.numberOfUsedColors(),
         round(sum(cgraph.getPayoffs()) / len(cgraph.getPayoffs()), 2), str(isgammanash), str(isnashboolean),
         round(gamma, 4), round(cgraph.getLLLThreshold(), 4), str(round(elap, 6)), str(round(its, 2)), epsilon,
         str(not cgraph.getNotLemmaValid())])


def graph_statistics(inpath, pattern):
    global NAMES
    global TYPES
    global SHORT_NAMES
    graph_dataset = []
    lll_dataset = []
    for l in [glob.glob(os.path.join(inpath, '*.nde')), glob.glob(os.path.join(inpath, '*.hist'))]:
        for filename in l:
            print(filename)
            # if "hist" not in filename:
            # raise Exception('computing stats on graphs, hist pattern expected')
            if filename.endswith('.nde'):
                graph = nk.nxadapter.nk2nx(nde2nk(filename))
                graph = graph.subgraph(max(nx.strongly_connected_components(graph), key=len))
                graph = nk.nxadapter.nx2nk(graph)
            else:
                graph = hist2nk(filename)
            keyname = filename.partition(".hist")[0] if filename.endswith('.hist') else filename.partition(".nde")[0]
            keyname = keyname.split("./")[1]
            degrees = [graph.degreeOut(i) for i in range(graph.numberOfNodes())]
            in_degrees = [graph.degreeIn(i) for i in range(graph.numberOfNodes())]
            const = round(findConst(graph), 2);
            lll_threshold = round(computeLLLThreshold(graph, 3), 2)  # change 3 if different K value
            graph_dataset.append(
                [keyname, SHORT_NAMES[keyname], SHORT_NAMES[keyname] + "ss", TYPES[keyname], graph.numberOfNodes(),
                 graph.numberOfEdges(), round(graph.numberOfEdges() / graph.numberOfNodes(), 2),
                 statistics.median(degrees), max(degrees),
                 "\\true{2pt}" if "random" in TYPES[keyname] else "\\false{2pt}",
                 "\\textbf{" + str(lll_threshold) + "}" if const >= 1 else lll_threshold]);
            # for k in [3,6,12,24,48]:
            #     lll_dataset.append([SHORT_NAMES[keyname]+"ss",k,round(statistics.mean(degrees),2),statistics.median(degrees),max(degrees),max(in_degrees),round(math.log(max(degrees)),2),round(math.log(max(in_degrees)),2),const,lll_threshold]);

    with open("graph_listing.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in graph_dataset:
            writer.writerow(i)

    with open("lll_listing.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in lll_dataset:
            writer.writerow(i)


def graph_analysis(inpath, pattern):
    global NAMES
    r = {}
    r['graph'] = []
    r['vertex'] = []
    r['degree'] = []
    r['LLL'] = []
    # average_degrees = {}
    median_degrees = {}
    for filename in glob.glob(os.path.join(inpath, pattern)):

        print("Degree Graph: " + filename)
        # print("Reading: "+filename)
        if filename.endswith('.nde'):
            graph = nk.nxadapter.nk2nx(nde2nk(filename))
            graph = graph.subgraph(max(nx.strongly_connected_components(graph), key=len))
            graph = nk.nxadapter.nx2nk(graph)
        elif filename.endswith('.hist'):
            graph = hist2nk(filename)
        # average_degrees[filename]=round(graph.numberOfEdges()/graph.numberOfNodes(),2)
        median_degrees[filename] = statistics.median([graph.degreeOut(i) for i in range(graph.numberOfNodes())])

    BLOCK_SIZE = 10;

    # print(sorted(median_degrees.items(), key=operator.itemgetter(1)))
    block_counter = 0;

    for filename, valore in sorted(median_degrees.items(), key=operator.itemgetter(1)):

        # for filename,valore in sorted(average_degrees.items(), key=operator.itemgetter(1)):

        # print("Reading: "+filename)
        if filename.endswith('.nde'):
            graph = nk.nxadapter.nk2nx(nde2nk(filename))
            graph = graph.subgraph(max(nx.strongly_connected_components(graph), key=len))
            graph = nk.nxadapter.nx2nk(graph)
            plotgraphname = filename.partition(".nde")[0]
            plotgraphname = plotgraphname.split("./")[1]
        elif filename.endswith('.hist'):
            graph = hist2nk(filename)
            plotgraphname = filename.partition(".hist")[0]
            plotgraphname = plotgraphname.split("./")[1]

        print("Graph: " + plotgraphname)

        maxindeg = 0;
        maxoutdeg = 0;
        for i in range(graph.numberOfNodes()):
            maxindeg = max(maxindeg, graph.degreeIn(i))
            maxoutdeg = max(maxoutdeg, graph.degreeOut(i))
        logprod = math.log(maxindeg) + math.log(maxoutdeg)
        cB = findConst(graph)

        LOG_THRESHOLD = 1;
        for i in range(graph.numberOfNodes()):
            r['graph'].append(NAMES[plotgraphname])
            r['vertex'].append(i)
            r['degree'].append(graph.degreeOut(i) + random.uniform(0.01, 0.1 + 0.1 * block_counter))
            # r['degree'].append(LOG_THRESHOLD+graph.degreeOut(i)+random.uniform(0.01,0.1+0.1*block_counter))
            if cB < 1:
                r['LLL'].append(-1)
            else:
                r['LLL'].append(cB * logprod)

        block_counter += 1;
        # median_width = 1.0
        if block_counter % BLOCK_SIZE == 0:  # dump and save block
            df = pd.DataFrame(data=r)
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (24.0, 10.0)

            plt.subplots_adjust(bottom=0.1, left=0.2)
            plt.rc('text', usetex=True)
            ax = sns.boxplot(x="graph", y="degree", data=df, whis=np.inf, width=.8)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
            # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%02d'))
            grafi = df['graph'];
            v = [i[-4:-1].upper() for i in grafi.unique()]
            name = "";
            for s in v:
                name += str(s) + "_"
            tmpg = [i for i in grafi.unique()]

            for i in range(len(ax.get_xticks())):
                if df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0] != -1:
                    ax.axhline(df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0], i * (1 / len(ax.get_xticks())),
                               i * (1 / len(ax.get_xticks())) + (1 / len(ax.get_xticks())), ls='dashdot', color='red',
                               linewidth=2)
                # ax.axvline(i,color='blue')
                else:
                    ax.axhline(LOG_THRESHOLD - 0.01, i * (1 / len(ax.get_xticks())),
                               i * (1 / len(ax.get_xticks())) + (1 / len(ax.get_xticks())), ls='dashed', color='blue',
                               linewidth=2)

            # for i in range(len(ax.get_xticks())):
            #     if df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0] >= 1:
            #         ax.axhline(df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0],i*(1/len(ax.get_xticks())),i*(1/len(ax.get_xticks()))+(1/len(ax.get_xticks())),ls='dashdot',color='red',linewidth=2)
            #     # ax.axvline(i,color='blue')
            #     else:
            #         ax.axhline(LOG_THRESHOLD-0.01,i*(1/len(ax.get_xticks())),i*(1/len(ax.get_xticks()))+(1/len(ax.get_xticks())),ls='dashed',color='blue',linewidth=2)

            plt.ylabel('\\textsc{degree}');
            plt.xlabel(None);
            plt.yscale('log')
            nnn = name + 'distrib.pdf'
            plt.gcf().set_size_inches(20, 8)

            print("Saving:", nnn)
            plt.savefig(nnn, bbox_inches='tight');
            destination = "/home/andrea/University/PhD/NashColoring/v6_results/whatis/"
            for filename in glob.glob(os.path.join("./", '*.pdf')):
                shutil.copy(filename, destination)
            r['graph'].clear()
            r['vertex'].clear()
            r['degree'].clear()
            r['LLL'].clear()

            # resized = False;

    if len(r['graph']) > 0:  ##dump remaining
        df = pd.DataFrame(data=r)
        plt.clf()
        sns.set(style="ticks", color_codes=True, font_scale=3)
        plt.rcParams['figure.figsize'] = (24.0, 10.0)
        plt.subplots_adjust(bottom=0.1, left=0.2)
        plt.rc('text', usetex=True)
        ax = sns.boxplot(x="graph", y="degree", data=df, whis=np.inf, width=.8)

        ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%02d'))
        grafi = df['graph'];
        v = [i[-4:-1].upper() for i in grafi.unique()]

        name = "";
        for s in v:
            name += str(s) + "_"

        tmpg = [i for i in grafi.unique()]
        # print(tmpg,[df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0] for i in range(BLOCK_SIZE)])
        for i in range(len(ax.get_xticks())):
            if df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0] != -1:
                ax.axhline(df.loc[df['graph'] == tmpg[i]]['LLL'].unique()[0], i * (1 / len(ax.get_xticks())),
                           i * (1 / len(ax.get_xticks())) + (1 / len(ax.get_xticks())), ls='dashdot', color='red',
                           linewidth=2)
            else:
                ax.axhline(LOG_THRESHOLD - 0.01, i * (1 / len(ax.get_xticks())),
                           i * (1 / len(ax.get_xticks())) + (1 / len(ax.get_xticks())), ls='dashed', color='blue',
                           linewidth=2)
        plt.ylabel('\\textsc{degree}');
        plt.xlabel(None);
        plt.yscale('log')
        # plt.yticks(None)

        nnn = name + 'distrib.pdf'
        # plt.rcParams['figure.figsize'] = (24.0, 8.0)
        plt.gcf().set_size_inches(20, 8)

        print("Saving:", nnn)
        plt.savefig(nnn, bbox_inches='tight');
        # destination="/home/andrea/University/PhD/NashColoring/v6_results/whatis/"
        # for filename in glob.glob(os.path.join("./", '*.pdf')):
        #     shutil.copy(filename, destination)
        r['graph'].clear()
        r['vertex'].clear()
        r['degree'].clear()
        r['LLL'].clear()


def ap1AlgoBarPlots(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)
        frames.append(df)
    resulting_df = pd.concat(frames)
    # print(resulting_df)

    resulting_df = resulting_df.replace(NAME[CODE['LLL']], '\\textsc{llg}')
    resulting_df = resulting_df.replace(NAME[CODE['ADY']], '\\textsc{br}')
    resulting_df = resulting_df.replace("DYN", '\\textsc{br}')
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)
    ordinamento = ["\\textsc{llg}", "\\textsc{br}"]

    # print(resulting_df)
    grafi = resulting_df['graph_name'];
    # print(resulting_df['k'].unique())

    # if len(grafi.unique())>1:
    #     raise Exception('pattern contains different graphs data')

    # print(grafi.unique())
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;
        # paletteset=sns.color_palette("dark").as_hex()
        # print(palet)
        paletteset = ["GnBu_r", "RdPu_r", "OrRd_r", "gist_gray_r", "BuGn_r", "Wistia"]

        # paletteset = ["bright","RdBu","Dark2","Greens_d"]
        for var in ['gamma', 'gamma-unhappy', 'nash-unhappy', 'avgpayoff', 'time', 'iterations']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95)
            plt.rc('text', usetex=True)
            k_values = {}
            # n_used = {}

            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                k_values[ordinamento.index(a)] = restr_df.query(q)['k'].unique()[0]
                # n_used[ordinamento.index(a)]=restr_df.query(q)['numused'].unique()[0]

            # print(k_values)
            rDF = pd.DataFrame(restr_df, columns=['algo', var])
            if var == "iterations":
                rDF = rDF[rDF['algo'] != "\\textsc{rnd}"]
                rDF = rDF[rDF['algo'] != "\\textsc{ap1}"]
                newdf = pd.melt(rDF, id_vars="algo", var_name="metrics", value_name="value")
                fig = sns.barplot(x="algo", y="value", palette=sns.color_palette(paletteset[ind]),
                                  order=["\\textsc{llg}", "\\textsc{br}"],
                                  hue="metrics", data=newdf)
            else:
                if var == 'gamma':
                    rDF['gamma'].values[rDF['gamma'].values > 10e6] = np.nan
                newdf = pd.melt(rDF, id_vars="algo", var_name="metrics", value_name="value")
                fig = sns.barplot(x="algo", y="value", palette=sns.color_palette(paletteset[ind]),
                                  order=ordinamento,
                                  hue="metrics", data=newdf)
            # plt.legend(loc=0,fontsize='12',handlelength=1,frameon=True)
            plt.legend(loc="lower left", mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize='24', handlelength=1,
                       fancybox=True, shadow=True)

            cou = 0
            maxH = 0.0

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))

            for p in fig.patches:
                # $$(used="+str(n_used[cou])+")$$
                fig.annotate("$$(k=" + str(k_values[cou]) + ")$$$$" + format(p.get_height(), '.2f') + "$$",
                             (p.get_x() + p.get_width() / 2., p.get_height() + maxH / 8.), ha='center', va='center',
                             xytext=(0, 10), textcoords='offset points', color="black", fontsize='x-small')
                cou += 1
            plt.ylim(0, maxH + (maxH / 3.))
            plt.xlabel(None);
            plt.ylabel(None);
            ind += 1;
            plt.gcf().set_size_inches(8, 6.5)
            filen = var + "_ap1_based_algo_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);

    best = {}
    second = {}
    for var in ['gamma', 'gamma-unhappy', 'nash-unhappy', 'avgpayoff', 'time', 'iterations']:
        best[var] = []
        second[var] = []
    for g in grafi.unique():
        q = "graph_name == \"" + str(g) + "\""
        restr_df = resulting_df.query(q)
        for var in ['gamma', 'gamma-unhappy', 'nash-unhappy', 'avgpayoff', 'time', 'iterations']:
            results = {}
            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                results[a] = restr_df.query(q)[var].mean()

            if var == 'avgpayoff':
                best[var].append(max(results, key=results.get))
                del results[max(results, key=results.get)]
                second[var].append(max(results, key=results.get))
                del results[max(results, key=results.get)]
            else:
                best[var].append(min(results, key=results.get))
                del results[min(results, key=results.get)]
                second[var].append(min(results, key=results.get))
                del results[min(results, key=results.get)]

    table = []
    text = ""
    text += "\\begin{table*}[thb]\n\\centering\n"
    for var in ['gamma', 'gamma-unhappy', 'nash-unhappy', 'avgpayoff', 'time', 'iterations']:
        head = True
        for a in algos:
            if head == True:
                # table.append((var,a,best[var].count(a),second[var].count(a),third[var].count(a),worst[var].count(a)))
                table.append(("\\textsc{" + var + "}", a,
                              str(best[var].count(a)) + " (" + str(
                                  round(100 * best[var].count(a) / len(best[var]), 1)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  100 * round(second[var].count(a) / len(second[var]), 1)) + " \\%)"))
                head = False
            else:
                table.append(("", a,
                              str(best[var].count(a)) + " (" + str(
                                  round(100 * best[var].count(a) / len(best[var]), 1)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  100 * round(second[var].count(a) / len(second[var]), 1)) + " \\%)"))
    from tabulate import tabulate
    text += tabulate(table, tablefmt='latex_raw',
                     headers=["\\textbf{Metric}", "\\textbf{algorithm}", "\\textbf{best}", "\\textbf{2nd}",
                              "\\textbf{total}"])
    text += "\n\\caption{Aggregate statistics for algorithms \\LLL, \\gre, respectively, when initialized with Ap1 coloring status, with respect to the four metrics, on all graphs.}"
    text += "\n\\label{table:ap1_algo:aggregate}"
    text += "\n\\end{table*}"

    print(text)


def multiAlgoBarPlots(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)
        if NAME[CODE['ALL']] not in filename:
            raise Exception('this is supposed to be plotting algo results')

        df = pd.read_csv(filename)
        frames.append(df)
    resulting_df = pd.concat(frames)
    # print(resulting_df)

    resulting_df = resulting_df.replace(NAME[CODE['LLL']], '\\textsc{llg}')
    resulting_df = resulting_df.replace(NAME[CODE['RND']], '\\textsc{rnd}')
    resulting_df = resulting_df.replace(NAME[CODE['AP1']], '\\textsc{ap1}')
    resulting_df = resulting_df.replace(NAME[CODE['AP3']], '\\textsc{ap3}')
    resulting_df = resulting_df.replace(NAME[CODE['ADY']], '\\textsc{mbr}')
    resulting_df = resulting_df.replace("DYN", '\\textsc{mbr}')
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)
    ordinamento = ["\\textsc{rnd}", "\\textsc{llg}", "\\textsc{ap1}", "\\textsc{mbr}"]

    print(resulting_df)
    # print(resulting_df)
    grafi = resulting_df['graph_name'];
    # print(resulting_df['k'].unique())

    # if len(grafi.unique())>1:
    #     raise Exception('pattern contains different graphs data')
    metric_to_latex = {'gamma': "\\ensuremath{\gamma(G,c)}", 'gamma-unhappy': "GU(G,c)",
                       'nash-unhappy': "\\ensuremath{U(G,c)}",
                       'avgpayoff': "\\ensuremath{\overline{P}(G,c)}", 'time': "\\ensuremath{T(G,c)}"}
    # print(grafi.unique())
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;
        # paletteset=sns.color_palette("dark").as_hex()
        # print(palet)
        paletteset = {metric_to_latex['gamma']: "#65312e", metric_to_latex['gamma-unhappy']: "#d95232",
                      metric_to_latex['nash-unhappy']: "#f29541", metric_to_latex['avgpayoff']: "#e3db9a",
                      metric_to_latex['time']: "#0B5650"}

        # paletteset = ["bright","RdBu","Dark2","Greens_d"]
        for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95)
            plt.rc('text', usetex=True)
            k_values = {}
            # n_used = {}

            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                k_values[ordinamento.index(a)] = restr_df.query(q)['k'].unique()[0]
                # n_used[ordinamento.index(a)]=restr_df.query(q)['numused'].unique()[0]

            # print(k_values)
            rDF = pd.DataFrame(restr_df, columns=['algo', var])

            if var == 'gamma':
                rDF['gamma'].values[rDF['gamma'].values > 10e6] = np.nan
            newdf = pd.melt(rDF, id_vars="algo", var_name="metrics", value_name="value")
            rand_row_min = newdf.loc[(newdf['algo'].str.contains('rnd'))]
            rand_row_min = rand_row_min[(rand_row_min['value'] == rand_row_min['value'].min())]
            newdf = newdf[~newdf['algo'].str.contains('rnd')]
            newdf = newdf.append(rand_row_min)
            newdf.reset_index(inplace=True)
            newdf.sort_index(inplace=True)

            fig = sns.barplot(x="algo", y="value", palette=paletteset,
                              order=ordinamento,
                              hue="metrics", data=newdf.replace(metric_to_latex))
            # plt.legend(loc=0,fontsize='12',handlelength=1,frameon=True)
            # plt.legend(loc="lower left",mode="expand", bbox_to_anchor=(0,1.02,1,0.2),fontsize='24',handlelength=1,fancybox=True, shadow=True)
            plt.legend([], [], frameon=False)
            cou = 0
            maxH = 0.0

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))

            for p in fig.patches:
                # $$(used="+str(n_used[cou])+")$$
                fig.annotate("$$" + format(p.get_height(), '.2f') + "$$",
                             (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10),
                             textcoords='offset points', color="black", fontsize='x-small')
                cou += 1
            if var == 'nash-unhappy':
                # plt.title('\\ensuremath{U(G,c)}',fontsize='small', horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{U(G,c)}', horizontalalignment='center', transform=fig.transAxes)
            if var == 'gamma':
                # plt.title('\\ensuremath{\gamma(G,c)}', fontsize='small',horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{\gamma(G,c)}', horizontalalignment='center', transform=fig.transAxes)
            if var == 'avgpayoff':
                # plt.title('\\ensuremath{\overline{P}(G,c)}', fontsize='small',horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{\overline{P}(G,c)}', horizontalalignment='center',
                         transform=fig.transAxes)

            if var == 'time':
                # plt.title('\\textsc{time}',fontsize='small', horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{T(G,c)}', horizontalalignment='center', transform=fig.transAxes)

            plt.ylim(0, maxH + (maxH / 3.))
            plt.xlabel(None);
            plt.ylabel(None);
            ind += 1;
            plt.gcf().set_size_inches(8, 6.5)
            filen = var + "_multi_algo_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);

    # destination="/home/andrea/University/PhD/NashColoring/v6_results/whatis/"
    # for filename in glob.glob(os.path.join("./", '*.pdf')):
    #     shutil.copy(filename, destination)

    best = {}
    second = {}
    third = {}
    worst = {}
    for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
        best[var] = []
        second[var] = []
        third[var] = []
        worst[var] = []
    for g in grafi.unique():
        q = "graph_name == \"" + str(g) + "\""
        restr_df = resulting_df.query(q)
        for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
            results = {}
            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                if 'rnd' in a:
                    results[a] = restr_df.query(q)[var].min()
                else:
                    results[a] = restr_df.query(q)[var].mean()

            if var == 'avgpayoff':
                best[var].append(max(results, key=results.get))
                del results[max(results, key=results.get)]
                second[var].append(max(results, key=results.get))
                del results[max(results, key=results.get)]
                third[var].append(max(results, key=results.get))
                del results[max(results, key=results.get)]
                assert (len(results) == 1)
                worst[var].append(max(results, key=results.get))
            else:
                best[var].append(min(results, key=results.get))
                del results[min(results, key=results.get)]
                second[var].append(min(results, key=results.get))
                del results[min(results, key=results.get)]
                third[var].append(min(results, key=results.get))
                del results[min(results, key=results.get)]
                assert (len(results) == 1)
                worst[var].append(min(results, key=results.get))

    table = []
    text = ""
    text += "\\begin{table*}[thb]\n\\centering\n"
    var_to_latex = {'gamma': "$\\gamma(G,c)$", 'gamma-unhappy': "$\\gunhappy(G,c)$", 'nash-unhappy': "$\\unhappy(G,c)$",
                    'avgpayoff': "$\\avgpayoff(G,c)$", 'time': "\\ensuremath{T(G,c)}"}
    algo_to_latex = {'\\textsc{llg}': "\\LLLshort", '\\textsc{ap1}': "\\apshort", '\\textsc{rnd}': "\\rndshort",
                     '\\textsc{mbr}': "\\greshort"}

    for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
        head = True
        for a in algos:
            if head == True:
                # table.append((var,a,best[var].count(a),second[var].count(a),third[var].count(a),worst[var].count(a)))
                table.append(("\\hline\\multirow{4}{*}{\\textsc{" + var_to_latex[var] + "}}", algo_to_latex[a],
                              str(best[var].count(a)) + " (" + str(
                                  round(100 * best[var].count(a) / len(best[var]), 0)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  100 * round(second[var].count(a) / len(second[var]), 0)) + " \\%)",
                              str(third[var].count(a)) + " (" + str(
                                  round(100 * third[var].count(a) / len(third[var]), 0)) + " \\%)",
                              str(worst[var].count(a)) + " (" + str(
                                  round(100 * worst[var].count(a) / len(worst[var]), 0)) + " \\%)",
                              str(len(worst[var])) + " (100 \\%)"))
                head = False
            else:
                table.append(("", algo_to_latex[a],
                              str(best[var].count(a)) + " (" + str(
                                  round(100 * best[var].count(a) / len(best[var]), 0)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  100 * round(second[var].count(a) / len(second[var]), 0)) + " \\%)",
                              str(third[var].count(a)) + " (" + str(
                                  round(100 * third[var].count(a) / len(third[var]), 0)) + " \\%)",
                              str(worst[var].count(a)) + " (" + str(
                                  round(100 * worst[var].count(a) / len(worst[var]), 0)) + " \\%)",
                              str(len(worst[var])) + " (100 \\%)"))
    from tabulate import tabulate
    text += tabulate(table, tablefmt='latex_raw',
                     headers=["\\textbf{metric}", "\\textbf{algorithm}", "\\textbf{best}", "\\textbf{2nd}",
                              "\\textbf{3rd}", "\\textbf{worst}", "\\textbf{total}"])
    text += "\n\\caption{Aggregate statistics for algorithms \\rnd, \\LLL, \\ap, \\gre, respectively, with respect to the four metrics, on all graphs.}"
    text += "\n\\label{table:multi_algo:aggregate}"
    text += "\n\\end{table*}"

    print(text)


def multiKLinePlot(inpath, pattern, sampling=True, sampling_interval=0.1):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        if NAME[CODE['DYN']] not in filename:
            raise Exception('this is supposed to be plotting dynamics datasets')
        print("Reading: " + filename)
        df = pd.read_csv(filename)

        assert len(df['avgdeg'].unique()) == 1
        assert len(df['maxdeg'].unique()) == 1
        frames.append(df)

    resulting_df = pd.concat(frames)
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)

    resulting_frames = []
    if sampling_interval > 1.0:
        raise Exception('sampling interval must be less than 100%')

    if len(resulting_df['iterations'].unique()) <= 2:
        raise Exception('dynamics with two iterations only')
    if sampling == True:
        print("Sampling")
        grafi = resulting_df;
        grafi = grafi['graph_name'];

        for graph_instance in grafi.unique():
            k_values = resulting_df
            k_values = k_values['k'];
            for kval in k_values.unique():
                max_iters = max(
                    resulting_df[(resulting_df['graph_name'] == graph_instance) & (resulting_df['k'] == kval)][
                        'iterations'].unique());
                sampling_points = round(max_iters * sampling_interval)
                resulting_frames.append(resulting_df[(resulting_df['k'] == kval) & (
                            resulting_df['graph_name'] == graph_instance) & (
                                                                 (resulting_df['iterations']) % sampling_points == 0)])
                resulting_frames.append(resulting_df[(resulting_df['k'] == kval) & (
                            resulting_df['graph_name'] == graph_instance) & (
                                                                 (resulting_df['iterations']) == max_iters)])

        resulting_df = pd.concat(resulting_frames)

    min_k = min(resulting_df['k'].unique())
    max_k = max(resulting_df['k'].unique())
    k_s = sorted([i for i in resulting_df['k'].unique()])
    # print(k_s)
    # WORKAROUND
    resulting_df["k"] = ["$%s$" % x for x in resulting_df["k"]]
    grafi = resulting_df;
    grafi = grafi['graph_name'];

    for graph_instance in grafi.unique():
        print("Processing graph: ", graph_instance)
        # print(resulting_df[resulting_df['graph_name']==graph_instance])

        assert len(resulting_df[resulting_df['graph_name'] == graph_instance]['avgdeg'].unique()) == 1
        assert len(resulting_df[resulting_df['graph_name'] == graph_instance]['maxdeg'].unique()) == 1
        for var in ['gamma', 'gammafocus', 'gamma-unhappy', 'nash-unhappy', 'avgpayoff']:
            # print("VAR",var)

            plt.clf()
            # plt.figure(figsize=(8,8))
            sns.set(style="ticks", color_codes=True, font_scale=3)
            # plt.rcParams.update({'figure.figsize':(8.0, 8.0)})
            plt.subplots_adjust(bottom=0.18, left=0.22)
            # plt.rcParams['figure.figsize'] = (18.0, 6.0)
            plt.rc('text', usetex=True)
            # plt.subplots_adjust(bottom=0.1,left=0.15)

            if var == 'gammafocus':
                if graph_instance == 'moreno_health.hist':
                    lll = []
                    with open('best_resp_202_moreno_health.hist_DYN_3_19_09_2022_13_56_06_936220.out', 'r') as f:
                        for l in f.readlines():
                            lll.append(float(l.strip()))
                    plt.plot(lll, 'b', linewidth=3.0)
                if graph_instance == 'dir_erdos_1000_0.2.hist':
                    lll = []
                    with open('dir_erdos_1000_0.2.hist_9965_12_02_2022_00_02_40_331687.out', 'r') as f:
                        for l in f.readlines():
                            lll.append(float(l.strip()))
                    plt.plot(lll, 'b', linewidth=3.0)
                if graph_instance == 'dir_erdos_1000_0.2.hist':
                    lll = []
                    with open('dir_erdos_1000_0.2.hist_9965_12_02_2022_00_02_40_331687.out', 'r') as f:
                        for l in f.readlines():
                            lll.append(float(l.strip()))
                    plt.plot(lll, 'b', linewidth=3.0)
                if graph_instance == 'oriented_email-Eu-core.hist':
                    lll = []
                    with open('oriented_email-Eu-core.hist_10022_14_02_2022_05_24_14_709252.out', 'r') as f:
                        for l in f.readlines():
                            x = float(l.strip())
                            if x > 10000:
                                x = 0
                            lll.append(x)
                    plt.plot(lll, 'b', linewidth=3.0)
                if graph_instance == 'oriented_oregon1_010331.hist':
                    lll = []
                    with open('oriented_oregon1_010331.hist_142778_14_02_2022_05_08_41_396920.out', 'r') as f:
                        for l in f.readlines():
                            x = float(l.strip())
                            if x > 10000:
                                x = 0
                            lll.append(x)
                    plt.plot(lll, 'b', linewidth=3.0)
                if graph_instance == 'twitter.hist':
                    lll = []
                    with open('twitter.hist_339154_12_02_2022_18_21_31_525560.out', 'r') as f:
                        for l in f.readlines():
                            x = float(l.strip())
                            if x > 10000:
                                x = 0
                            lll.append(x)
                    plt.plot(lll, 'b', linewidth=3.0)
                g = sns.lineplot(x='iterations', y='gamma',
                                 data=resulting_df[resulting_df['graph_name'] == graph_instance],
                                 hue_order=["$%s$" % x for x in k_s], alpha=0.9, linewidth=4, markers=True,
                                 color='#D32F2F')
            else:
                g = sns.lineplot(x='iterations', y=var, data=resulting_df[resulting_df['graph_name'] == graph_instance],
                                 hue_order=["$%s$" % x for x in k_s], alpha=0.9, linewidth=4, markers=True,
                                 color='#D32F2F')

            if var == 'gammafocus':
                plt.ylabel(' ');
                g.axhline(1, ls='dashdot', color='black', label="Pure NE")

            if var == 'gamma-unhappy':
                plt.ylabel('\\ensuremath{\overline{U}(G,c)}');
                # g.axhline(0, ls='dashdot',color='orange',label="Gamma-Unhappy")

            if var == 'nash-unhappy':
                plt.ylabel('\\ensuremath{\overline{U}(G,c)}');
                # g.axhline(0, ls='dashdot',color='blue',label="Nash-Unhappy")

            if var == 'gamma':
                plt.ylabel('\\ensuremath{\gamma(G,c)}');
                g.axhline(1, ls='dashdot', color='black', label="Pure NE")
                plt.plot(frames[0]['iterations'], frames[0]['gamma'], '#D32F2F', linewidth=3.0)
                print(frames[0]['gamma'])

                # g.axhline(1, ls='-.',color='blue',label="Nash")
                # g.axhline(resulting_df[resulting_df['graph_name']==graph_instance]['avgdeg'].unique()[0], ls='--',color='red',label="AvgDeg")
                # g.axhline(resulting_df[resulting_df['graph_name']==graph_instance]['maxdeg'].unique()[0], ls='dashdot',color='black',label="MaxDeg")
            if var == 'avgpayoff':
                plt.ylabel('\\ensuremath{\overline{P}(G,c)}');
                # g.axhline(resulting_df[resulting_df['graph_name']==graph_instance]['avgdeg'].unique()[0], ls='--',color='red',label="AvgDeg")

            plt.xlabel('\\textsc{iterations}');
            plt.gcf().set_size_inches(8, 6.3)
            plt.legend(loc="lower left", mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), fancybox=True, shadow=True,
                       ncol=4)
            # current_handles, current_labels = plt.gca().get_legend_handles_labels()
            # current_handles.pop(0)
            # current_labels.pop(0)
            # plt.legend(current_handles,current_labels,loc="lower left",mode="expand", bbox_to_anchor=(0,1.02,1,0.2),handlelength=1,fancybox=True, shadow=True, ncol=3)
            plt.subplots_adjust(top=0.7)
            stringoutput = var + "_multi_k_" + str(min_k) + "_" + str(max_k) + "_" + graph_instance.partition(".hist")[
                0] + ".pdf"
            print("Saving: " + stringoutput)
            plt.savefig(stringoutput);

        destination = "/home/andrea/University/PhD/NashColoring/v6_results/whatis/"
        for filename in glob.glob(os.path.join("./", '*.pdf')):
            shutil.copy(filename, destination)


def multiEpsilonBarPlots(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        if NAME[CODE['AP3']] not in filename:
            raise Exception('this is supposed to be plotting ap3 datasets')
        print("Reading: " + filename)
        df = pd.read_csv(filename)
        assert len(df['avgdeg'].unique()) == 1
        assert len(df['maxdeg'].unique()) == 1
        frames.append(df)

    resulting_df = pd.concat(frames)
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)

    # min_k = min(resulting_df['k'].unique())
    # max_k = max(resulting_df['k'].unique())
    # k_s = sorted([i for i in resulting_df['k'].unique()])
    # # print(k_s)
    # #WORKAROUND
    # resulting_df["k"] = ["$%s$" % x for x in resulting_df["k"]]
    grafi = resulting_df['graph_name'];

    # if 1 not in resulting_df['iterations'].unique() or len(resulting_df['iterations'].unique())>1:
    # raise Exception('iterations larger than one for ap3')

    # print(grafi.unique())
    # if len(grafi.unique())>1:
    #     raise Exception('pattern contains different graphs data')
    metric_to_latex = {'gamma': "\\ensuremath{\gamma(G,c)}", 'gamma-unhappy': "GU(G,c)",
                       'nash-unhappy': "\\ensuremath{U(G,c)}",
                       'avgpayoff': "\\ensuremath{\overline{P}(G,c)}", 'time': "\\textsc{time}"}
    # print(grafi.unique())
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;

        paletteset = {metric_to_latex['gamma']: "#65312e", metric_to_latex['gamma-unhappy']: "#d95232",
                      metric_to_latex['nash-unhappy']: "#f29541", metric_to_latex['avgpayoff']: "#e3db9a",
                      metric_to_latex['time']: "#0B5650"}

        # paletteset = ["bright","RdBu","Dark2","Greens_d"]

        for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95)

            plt.rc('text', usetex=True)
            k_values = {}
            used_values = {}

            epsilons = restr_df['epsilon'].unique()
            for e in epsilons:
                # q = "epsilon == \"\\"+str(e)+"\""
                q = "epsilon == " + str(e)
                k_values[e] = restr_df.query(q)['k'].unique()[0]
                used_values[e] = restr_df.query(q)['numused'].unique()[0]

            # print(epsilons,k_values)
            rDF = pd.DataFrame(restr_df, columns=['epsilon', var])
            if var == 'gamma':
                rDF['gamma'].values[rDF['gamma'].values > 10e6] = np.nan
            # print(rDF)
            newdf = pd.melt(rDF, id_vars="epsilon", var_name="metrics", value_name="value")
            # print(newdf)
            fig = sns.barplot(x="epsilon", y="value", palette=paletteset,
                              hue="metrics", data=newdf.replace(metric_to_latex))
            # plt.legend(loc=0,fontsize='12',handlelength=1,frameon=True)
            plt.legend(loc="lower left", mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize='24', handlelength=1,
                       fancybox=True, shadow=True)

            plt.gcf().set_size_inches(8, 6.5)

            cou = 0
            maxH = 0.0

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))

            for p in fig.patches:
                #                fig.annotate("\\textbf{$$(k="+str(k_values[epsilons[cou]])+")$$$$"+format(p.get_height(), '.2f')+"$$}", (p.get_x() + p.get_width() / 2., p.get_height()+maxH/8.), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points',color="black",fontsize='x-small')
                fig.annotate("$$(k=" + str(k_values[epsilons[cou]]) + ")$$$$(used=" + str(
                    used_values[epsilons[cou]]) + ")$$$$" + format(p.get_height(), '.2f') + "$$",
                             (p.get_x() + p.get_width() / 2., p.get_height() + maxH / 8.), ha='center', va='center',
                             xytext=(0, 10), textcoords='offset points', color="black", fontsize='xx-small')
                cou += 1

            if maxH != 0.0:
                plt.ylim(0, maxH + maxH / 3.)

            plt.xlabel(None);
            plt.ylabel(None);
            ind += 1;

            filen = var + "_multi_epsilon_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);
    destination = "/home/andrea/University/PhD/NashColoring/v6_results/whatis/"
    for filename in glob.glob(os.path.join("./", '*.pdf')):
        shutil.copy(filename, destination)


def ap3_vs_allBarPlots(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        if NAME[CODE['AV3']] not in filename:
            raise Exception('this is supposed to be plotting ap3_vs_ap1 datasets')
        print("Reading: " + filename)
        df = pd.read_csv(filename)
        assert len(df['avgdeg'].unique()) == 1
        assert len(df['maxdeg'].unique()) == 1
        frames.append(df)

    resulting_df = pd.concat(frames)

    grafi = resulting_df['graph_name'];
    resulting_df = resulting_df.replace(NAME[CODE['LLL']], '\\textsc{llg}')
    resulting_df = resulting_df.replace(NAME[CODE['RND']], '\\textsc{rnd}')
    resulting_df = resulting_df.replace(NAME[CODE['AP1']], '\\textsc{ap1}')
    resulting_df = resulting_df.replace(NAME[CODE['AP3']], '\\textsc{ap3}')
    resulting_df = resulting_df.replace(NAME[CODE['ADY']], '\\textsc{mbr}')
    resulting_df = resulting_df.replace("DYN", '\\textsc{mbr}')
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)

    ordinamento = ["\\textsc{llg}", "\\textsc{ap1}", "\\textsc{mbr}", "\\textsc{ap3}"]

    metric_to_latex = {'gamma': "\\ensuremath{\gamma(G,c)}", 'gamma-unhappy': "GU(G,c)",
                       'nash-unhappy': "\\ensuremath{U(G,c)}",
                       'avgpayoff': "\\ensuremath{\overline{P}(G,c)}", 'time': "\\textsc{time}"}
    epsilon = resulting_df['epsilon'].unique()[0]
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;  # fixed

        paletteset = {metric_to_latex['gamma']: "#65312e", metric_to_latex['gamma-unhappy']: "#d95232",
                      metric_to_latex['nash-unhappy']: "#f29541", metric_to_latex['avgpayoff']: "#e3db9a",
                      metric_to_latex['time']: "#0B5650"}

        # paletteset = ["bright","RdBu","Dark2","Greens_d"]

        for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15)
            plt.rc('text', usetex=True)

            k_values = {}
            used_values = {}
            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                k_values[ordinamento.index(a)] = restr_df.query(q)['numused'].unique()[0]

            rDF = pd.DataFrame(restr_df, columns=['algo', var])
            newdf = pd.melt(rDF, id_vars="algo", var_name="metrics", value_name="value")
            # fig=sns.barplot(x="algo", y="value",palette=sns.color_palette(paletteset[ind]), hue="metrics", data=newdf)

            fig = sns.barplot(x="algo", y="value", palette=paletteset,
                              hue="metrics", data=newdf.replace(metric_to_latex))
            plt.legend(loc="lower left", mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), fontsize='24', handlelength=1,
                       fancybox=True, shadow=True, ncol=4)
            plt.legend([], [], frameon=False)

            cou = 0
            maxH = 0.0

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))

            for p in fig.patches:
                # $$(used="+str(n_used[cou])+")$$
                fig.annotate("$$" + format(p.get_height(), '.2f') + "$$",
                             (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10),
                             textcoords='offset points', color="black", fontsize='x-small')
                cou += 1
            fig.text(0.1, .9, '$\\epsilon=' + str(epsilon) + '$', horizontalalignment='center', transform=fig.transAxes,
                     fontsize='xx-small')
            fig.text(0.9, .9, '$k=' + str(k_values[0]) + '$', horizontalalignment='center', transform=fig.transAxes,
                     fontsize='xx-small')

            if var == 'nash-unhappy':
                # plt.title('\\ensuremath{U(G,c)}',fontsize='small', horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{U(G,c)}', horizontalalignment='center', transform=fig.transAxes)
            if var == 'gamma':
                # plt.title('\\ensuremath{\gamma(G,c)}', fontsize='small',horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{\gamma(G,c)}', horizontalalignment='center', transform=fig.transAxes)
            if var == 'avgpayoff':
                # plt.title('\\ensuremath{\overline{P}(G,c)}', fontsize='small',horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{\overline{P}(G,c)}', horizontalalignment='center',
                         transform=fig.transAxes)

            if var == 'time':
                # plt.title('\\textsc{time}',fontsize='small', horizontalalignment='center');
                fig.text(.5, .9, '\\ensuremath{T(G,c)}', horizontalalignment='center', transform=fig.transAxes)

            plt.ylim(0, maxH + maxH / 3.)
            plt.xlabel(None);
            plt.ylabel(None);
            ind += 1;

            plt.gcf().set_size_inches(8, 6.5)

            filen = var + "_ap3_vs_all_" + str(epsilon) + "_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);


def multiAlgomultiKBarPlots(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)
        if NAME[CODE['ALL']] not in filename:
            raise Exception('this is supposed to be plotting algo results')

        df = pd.read_csv(filename)
        frames.append(df)
    resulting_df = pd.concat(frames)
    # print(resulting_df)

    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)
    resulting_df = resulting_df.replace(NAME[CODE['LLL']], '\\textsc{llg}')
    resulting_df = resulting_df.replace(NAME[CODE['RND']], '\\textsc{rnd}')
    resulting_df = resulting_df.replace(NAME[CODE['AP1']], '\\textsc{ap1}')
    resulting_df = resulting_df.replace(NAME[CODE['AP3']], '\\textsc{ap3}')
    resulting_df = resulting_df.replace(NAME[CODE['ADY']], '\\textsc{mbr}')
    resulting_df = resulting_df.replace('DYN', '\\textsc{mbr}')
    ordinamento = ["\\textsc{rnd}", "\\textsc{llg}", "\\textsc{ap1}", "\\textsc{mbr}"]

    # print(resulting_df)
    grafi = resulting_df['graph_name'];
    # print(resulting_df['k'].unique())

    # if len(grafi.unique())>1:
    #     raise Exception('pattern contains different graphs data')

    # print(grafi.unique())
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;
        # paletteset=sns.color_palette("dark").as_hex()
        # print(palet)
        paletteset = ["BrBG_r", "BuPu_r", "CMRmap_r", "OrRd_r"]
        for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95)
            if var == 'nash-unhappy' or var == 'avgpayoff':
                plt.subplots_adjust(bottom=0.1, left=0.2, right=0.95)

            plt.rc('text', usetex=True)
            # k_values = {}
            # algos = restr_df['algo'].unique()
            # for a in algos:
            #     q = "algo == \"\\"+a+"\""
            #     k_values[ordinamento.index(a)]=restr_df.query(q)['k'].unique()[0]

            # print(k_values)
            rDF = pd.DataFrame(restr_df, columns=['algo', 'k', var])
            if var == 'gamma':
                rDF['gamma'].values[rDF['gamma'].values > 10e6] = np.nan
            newdf = pd.melt(rDF, id_vars=["algo", 'k'], var_name="metrics", value_name="value")
            # print(newdf)
            rand_row_min = newdf.loc[(newdf['algo'].str.contains('rnd'))]
            rand_rows = []
            for kappa in rand_row_min['k'].unique():
                temp_k = rand_row_min.loc[rand_row_min['k'] == kappa]
                if np.isnan(temp_k['value'].unique()[0]):
                    rand_rows.append(temp_k.iloc[0])
                else:
                    rand_rows.append(temp_k[(temp_k['value'] == temp_k['value'].min())])
            newdf = newdf[~newdf['algo'].str.contains('rnd')]
            for rand_row in rand_rows:
                newdf = newdf.append(rand_row)
            newdf.reset_index(inplace=True)
            newdf.sort_index(inplace=True)
            sns.barplot(x="algo", y="value", palette=sns.color_palette(paletteset[ind], n_colors=7),
                        order=ordinamento, hue="k", data=newdf)
            plt.legend(loc="lower left", bbox_to_anchor=(0, 1.04, 1, 1), fontsize='18', ncol=len(newdf['k'].unique()),
                       mode="expand", handlelength=1, fancybox=True, shadow=True)
            # plt.legend(loc="lower left",mode="expand",fontsize='24', ncol = 5)
            if var == 'nash-unhappy':
                plt.title('\\ensuremath{U(G,c)}', loc='left', fontsize='small', horizontalalignment='right');
            if var == 'gamma':
                plt.title('\\ensuremath{\gamma(G,c)}', loc='left', fontsize='small', horizontalalignment='right');
            if var == 'avgpayoff':
                plt.title('\\ensuremath{\overline{P}(G,c)}', loc='left', fontsize='small', horizontalalignment='right');
            if var == 'time':
                plt.title('\\ensuremath{T(G,c)}', loc='left', fontsize='small', horizontalalignment='right');
            if var == 'iterations':
                plt.title('\\textsc{iterations}', loc='left', fontsize='small', horizontalalignment='right');
            plt.xlabel(None);
            plt.ylabel(None);
            ind += 1;

            plt.gcf().set_size_inches(8, 6.5)

            filen = var + "_multi_algo_multi_k_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);

    # destination="/home/andrea/University/PhD/NashColoring/v6_results/whatis/"

    # for filename in glob.glob(os.path.join("./", '*.pdf')):
    #     shutil.copy(filename, destination)

    best = {}
    second = {}
    third = {}
    worst = {}
    for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
        best[var] = []
        second[var] = []
        third[var] = []
        worst[var] = []
    for g in grafi.unique():
        q = "graph_name == \"" + str(g) + "\""
        restr_df = resulting_df.query(q)
        for k_val in restr_df['k'].unique():
            q = "k == " + str(k_val)

            restr_restr_df = restr_df.query(q)

            for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:

                results = {}
                algos = restr_restr_df['algo'].unique()
                for a in algos:
                    q = "algo == \"\\" + a + "\""
                    if 'rnd' in a:
                        results[a] = restr_df.query(q)[var].min()
                    else:
                        results[a] = restr_df.query(q)[var].mean()

                if var == 'avgpayoff':
                    best[var].append(max(results, key=results.get))
                    del results[max(results, key=results.get)]
                    second[var].append(max(results, key=results.get))
                    del results[max(results, key=results.get)]
                    third[var].append(max(results, key=results.get))
                    del results[max(results, key=results.get)]
                    assert (len(results) == 1)
                    worst[var].append(max(results, key=results.get))
                else:
                    best[var].append(min(results, key=results.get))
                    del results[min(results, key=results.get)]
                    second[var].append(min(results, key=results.get))
                    del results[min(results, key=results.get)]
                    third[var].append(min(results, key=results.get))
                    del results[min(results, key=results.get)]
                    assert (len(results) == 1)
                    worst[var].append(min(results, key=results.get))
    table = []
    text = ""
    text += "\\begin{table*}[thb]\n\\centering\n"
    var_to_latex = {'gamma': "$\\gamma(G,c)$", 'gamma-unhappy': "$\\gunhappy(G,c)$", 'nash-unhappy': "$\\unhappy(G,c)$",
                    'avgpayoff': "$\\avgpayoff(G,c)$", 'time': "\\ensuremath{T(G,c)}"}
    algo_to_latex = {'\\textsc{llg}': "\\LLLshort", '\\textsc{ap1}': "\\apshort", '\\textsc{rnd}': "\\rndshort",
                     '\\textsc{mbr}': "\\greshort"}
    for var in ['gamma', 'nash-unhappy', 'avgpayoff', 'time']:
        head = True
        for a in algos:
            if head == True:
                # table.append((var,a,best[var].count(a),second[var].count(a),third[var].count(a),worst[var].count(a)))
                table.append(("\\hline\\multirow{4}{*}{\\textsc{" + var_to_latex[var] + "}}", algo_to_latex[a],
                              str(best[var].count(a)) + " (" + str(round(100 * best[var].count(a) / 210, 1)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  round(100 * second[var].count(a) / 210, 1)) + " \\%)",
                              str(third[var].count(a)) + " (" + str(
                                  round(100 * third[var].count(a) / 210, 1)) + " \\%)",
                              str(worst[var].count(a)) + " (" + str(
                                  round(100 * worst[var].count(a) / 210, 1)) + " \\%)",
                              str(len(worst[var])) + " (100 \\%)"))
                head = False
            else:
                table.append(("", algo_to_latex[a],
                              str(best[var].count(a)) + " (" + str(round(100 * best[var].count(a) / 210, 1)) + " \\%)",
                              str(second[var].count(a)) + " (" + str(
                                  round(100 * second[var].count(a) / 210, 1)) + " \\%)",
                              str(third[var].count(a)) + " (" + str(
                                  round(100 * third[var].count(a) / 210, 1)) + " \\%)",
                              str(worst[var].count(a)) + " (" + str(
                                  round(100 * worst[var].count(a) / 210, 1)) + " \\%)",
                              str(len(worst[var])) + " (100 \\%)"))
    from tabulate import tabulate
    text += tabulate(table, tablefmt='latex_raw',
                     headers=["\\textbf{metric}", "\\textbf{algorithm}", "\\textbf{best}", "\\textbf{2nd}",
                              "\\textbf{3rd}", "\\textbf{worst}", "\\textbf{total}"])
    text += "\n\\caption{Aggregate statistics for algorithms \\rndshort, \\LLLshort, \\apshort, \\greshort, respectively, with respect to the four metrics, on all graphs, for values of $k$ > 3.}"
    text += "\n\\label{table:aggregate:multi_k}"
    text += "\n\\end{table*}"

    print(text)


def plotUnhappyTrend(graph_name, iteration, trend):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
    f = plt.figure(figsize=(20, 20))
    plt.plot(range(len(trend)), trend)
    plt.xlabel("Iteration")
    plt.ylabel("Number of unhappy nodes")
    plt.grid()
    f.savefig(graph_name + "_" + str(iteration) + "_" + date_time + ".jpg")
    trend_file = open(graph_name + "_" + str(iteration) + "_" + date_time + ".out", 'w+')
    for t in trend:
        trend_file.write(str(t) + "\n")
    trend_file.close()


def rand_vs_lll_vs_ap1(inpath, pattern):
    frames = []
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)
        if NAME[CODE['ALL']] not in filename:
            raise Exception('this is supposed to be plotting algo results')

        df = pd.read_csv(filename)
        frames.append(df)
    resulting_df = pd.concat(frames)
    # print(resulting_df)

    metric_to_latex = {'gamma': "\\ensuremath{\gamma(G,c)}", 'gamma-unhappy': "GU(G,c)",
                       'nash-unhappy': "\\ensuremath{U(G,c)}",
                       'avgpayoff': "\\ensuremath{\overline{P}(G,c)}", 'time': "\\textsc{time}"}

    resulting_df = resulting_df.replace(NAME[CODE['LLL']], '\\textsc{llg}')
    resulting_df = resulting_df.replace(NAME[CODE['RND']], '\\textsc{rnd}')
    resulting_df = resulting_df.replace(NAME[CODE['AP1']], '\\textsc{ap1}')
    resulting_df = resulting_df.replace(NAME[CODE['AP3']], '\\textsc{ap3}')
    resulting_df = resulting_df.replace(NAME[CODE['ADY']], '\\textsc{br}')
    resulting_df = resulting_df.replace("DYN", '\\textsc{br}')
    resulting_df = resulting_df[~resulting_df['algo'].str.contains('br')]
    resulting_df.rename(columns={'gamma_unhappy': "gamma-unhappy", "nash_unhappy": "nash-unhappy"}, inplace=True)
    ordinamento = ["\\textsc{rnd}", "\\textsc{llg}", "\\textsc{ap1}"]

    print(resulting_df['algo'])
    # print(resulting_df)
    grafi = resulting_df['graph_name'];
    # print(resulting_df['k'].unique())
    resulting_df["gamma-unhappy"] = resulting_df["gamma-unhappy"].round(2)

    # if len(grafi.unique())>1:
    #     raise Exception('pattern contains different graphs data')

    # print(grafi.unique())
    for g in grafi.unique():
        print("GRAPH:", str(g))
        q = "graph_name == \"" + str(g) + "\""
        # print(q)

        restr_df = resulting_df.query(q)
        ind = 0;
        # paletteset=sns.color_palette("dark").as_hex()
        # print(palet)
        paletteset = {metric_to_latex['gamma']: "#65312e", metric_to_latex['gamma-unhappy']: "#d95232",
                      metric_to_latex['nash-unhappy']: "#f29541", metric_to_latex['avgpayoff']: "#e3db9a",
                      metric_to_latex['time']: "#0B5650"}

        # paletteset = ["bright","RdBu","Dark2","Greens_d"]
        for var in ['gamma']:
            plt.clf()
            sns.set(style="ticks", color_codes=True, font_scale=3)
            plt.rcParams['figure.figsize'] = (12.0, 6.0)

            plt.subplots_adjust(bottom=0.1, left=0.15, right=0.95)
            plt.rc('text', usetex=True)
            k_values = {}
            # n_used = {}

            algos = restr_df['algo'].unique()
            for a in algos:
                q = "algo == \"\\" + a + "\""
                k_values[ordinamento.index(a)] = restr_df.query(q)['k'].unique()[0]
                # n_used[ordinamento.index(a)]=restr_df.query(q)['numused'].unique()[0]

            # print(k_values)
            rDF = pd.DataFrame(restr_df, columns=['algo', var])

            if var == 'gamma':
                plt.title('\\ensuremath{\gamma(G,c)}', loc='center', fontsize='small', horizontalalignment='right');
                rDF['gamma'].values[rDF['gamma'].values > 10e6] = np.nan
            newdf = pd.melt(rDF, id_vars="algo", var_name="metrics", value_name="value")
            rand_row_min = newdf.loc[(newdf['algo'].str.contains('rnd'))]
            rand_row_min = rand_row_min[(rand_row_min['value'] == rand_row_min['value'].min())]
            newdf = newdf[~newdf['algo'].str.contains('rnd')]
            newdf = newdf.append(rand_row_min)
            newdf.reset_index(inplace=True)
            newdf.sort_index(inplace=True)

            fig = sns.barplot(x="algo", y="value", palette=paletteset,
                              order=ordinamento,
                              hue="metrics", data=newdf.replace(metric_to_latex))
            # plt.legend(loc=0,fontsize='12',handlelength=1,frameon=True)
            # plt.legend(loc="lower left",mode="expand", bbox_to_anchor=(0,1.02,1,0.2),fontsize='24',handlelength=1,fancybox=True, shadow=True)
            cou = 0
            maxH = 0.0

            for p in fig.patches:
                maxH = max(maxH, float(p.get_height()))
            print(k_values)
            for p in fig.patches:
                # $$(used="+str(n_used[cou])+")$$
                fig.annotate("$$(k=" + str(k_values[cou]) + ")$$$$" + format(p.get_height(), '.2f') + "$$",
                             (p.get_x() + p.get_width() / 2., p.get_height() + maxH / 8.), ha='center', va='center',
                             xytext=(0, 10), textcoords='offset points', color="black", fontsize='x-small')
                cou += 1
            plt.ylim(0, maxH + (maxH / 3.))
            plt.xlabel(None);
            plt.ylabel(None);
            plt.gca().get_legend().remove()
            ind += 1;
            plt.gcf().set_size_inches(8, 6.5)
            filen = var + "_rand_vs_lll_vs_ap1_" + str(g).partition(".hist")[0] + ".pdf"
            print("Saving: " + filen)
            plt.savefig(filen);


def extract_best_gamma(inpath, pattern):
    frames = []
    resulting_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)
        best_gamma_row = df[df['gamma'] == df['gamma'].min()].iloc[0]
        best_gamma_row['time'] = df.iloc[-1]['time']
        best_gamma_row['iterations'] = df.iloc[-1]['iterations']

        resulting_df = resulting_df.append(best_gamma_row)
    resulting_df = resulting_df.reset_index().drop('index', axis=1)
    resulting_df.to_csv(resulting_df['graph_name'][0] + '_DYN_3.csv', index=False)


def extract_best_lll(inpath, pattern):
    frames = []
    resulting_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)
        best_gamma_row = df[df['gamma'] == df['gamma'].min()].iloc[0]
        best_gamma_row['time'] = df.iloc[-1]['time']
        best_gamma_row['iterations'] = df.iloc[-1]['iterations']

        resulting_df = resulting_df.append(best_gamma_row)
    resulting_df = resulting_df.reset_index().drop('index', axis=1)
    resulting_df.to_csv(resulting_df['graph_name'][0] + '_LLL_3.csv', index=False)


def collect_rnd(inpath, pattern):
    frames = []
    resulting_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)

        resulting_df = resulting_df.append(df.iloc[0])
    resulting_df = resulting_df.reset_index().drop('index', axis=1)
    resulting_df.to_csv(resulting_df['graph_name'][0] + '_RND_3.csv', index=False)


def collect_rnd_multi_k(inpath, pattern):
    frames = []
    agglomerated_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)

        agglomerated_df = agglomerated_df.append(df.iloc[0])
    for k in agglomerated_df['k'].unique():
        temp_df = agglomerated_df.loc[agglomerated_df['k'] == k]
        temp_df.to_csv(temp_df.iloc[0]['graph_name'] + '_RND_multi_k_' + str(k) + '.csv', index=False)


def collect_multi_k_mbr(inpath, pattern):
    frames = []
    agglomerated_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)
        best_gamma_row = df[df['gamma'] == df['gamma'].min()].iloc[0]
        best_gamma_row['time'] = df.iloc[-1]['time']
        best_gamma_row['iterations'] = df.iloc[-1]['iterations']
        agglomerated_df = agglomerated_df.append(best_gamma_row)
    for k in agglomerated_df['k'].unique():
        temp_df = agglomerated_df.loc[agglomerated_df['k'] == k]
        temp_df.to_csv(temp_df.iloc[0]['graph_name'] + '_DYN_multi_k_' + str(k) + '.csv', index=False)


def collect_multi_k_lll(inpath, pattern):
    frames = []
    agglomerated_df = pd.DataFrame()
    for filename in glob.glob(os.path.join(inpath, pattern)):
        print("Reading: " + filename)

        df = pd.read_csv(filename)
        if df.size == 0:
            continue
        best_gamma_row = df[df['gamma'] == df['gamma'].min()].iloc[0]
        best_gamma_row['time'] = df.iloc[-1]['time']
        best_gamma_row['iterations'] = df.iloc[-1]['iterations']
        agglomerated_df = agglomerated_df.append(best_gamma_row)
    for k in agglomerated_df['k'].unique():
        temp_df = agglomerated_df.loc[agglomerated_df['k'] == k]
        temp_df.to_csv(temp_df.iloc[0]['graph_name'] + '_LLL_multi_k_' + str(k) + '.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KColoring Plotter')
    parser.add_argument("--p", metavar="PATTERN or FILE", required=True, default="",
                        help="pattern to csv files (or single file)")
    parser.add_argument("--t", metavar="PLOT_TYPE", required=True, default=0,
                        help="plot_Type [0: ap3 various eps - 1: dynamics multi_k - 2: multi_algo single_k - 3: graph_analysis - 4: ap3_vs_ap1 - 5: multiAlgo-multiK - 6: graph_statistics (includes LLL) - 7: Ap1 based]")
    parser.add_argument("--n", metavar="NEW_GRAPH_NAME", required=False, default='',
                        help="Name of the file which will contain the cleared graph by nk")

    # parser.add_argument("--k",metavar="NUM_COLORS", required=False, default=3)

    args = parser.parse_args()

    if str(args.p) == "":
        raise Exception('wrong pattern')
    print("PATTERN or FILE:", str(args.p))
    print("WARNING -- USE QUOTES FOR PATTERNS")
    print("PLOT_TYPE:", int(args.t))
    if (int(args.t) not in range(16)):
        raise Exception('wrong args')

    if (int(args.t) == 0):
        multiEpsilonBarPlots('./', str(args.p))
    if (int(args.t) == 1):
        multiKLinePlot('./', str(args.p), True, 0.1)
    if (int(args.t) == 2):
        multiAlgoBarPlots('./', str(args.p))
    if (int(args.t) == 3):
        graph_analysis('./', str(args.p))
    if (int(args.t) == 4):
        ap3_vs_allBarPlots('./', str(args.p))
    if (int(args.t) == 5):
        multiAlgomultiKBarPlots('./', str(args.p))
    if (int(args.t) == 6):
        graph_statistics('./', str(args.p))
    if (int(args.t) == 7):
        ap1AlgoBarPlots('./', str(args.p))
    if (int(args.t) == 8):
        ntwk2hist(args.n, hist2nk(args.p))
    if (int(args.t) == 9):
        rand_vs_lll_vs_ap1('./', str(args.p))
    if (int(args.t) == 10):
        extract_best_gamma('./', str(args.p))
    if (int(args.t) == 11):
        extract_best_lll('./', str(args.p))
    if (int(args.t) == 12):
        collect_rnd('./', str(args.p))
    if (int(args.t) == 13):
        collect_multi_k_mbr('./', str(args.p))
    if (int(args.t) == 14):
        collect_multi_k_lll('./', str(args.p))
    if (int(args.t) == 15):
        collect_rnd_multi_k('./', str(args.p))
