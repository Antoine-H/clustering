# -*- coding: utf-8 -*-
#!/usr/bin/python3
#
#
# Implementation of a 2-approximation algorithm for the fully dynamic k-center
# clustering problem.
# From https://sites.google.com/site/maurosozio/techReport.pdf
#


import random
import operator
import math
import cartopy.crs       as ccrs
import matplotlib.pyplot as plt
import timeit
import sys


# Reads dataset, stores tweets as [timestamp, latitude, longitude].
def read_tweets (file):
    tweets = []
    with open(file) as input:
        for line in input:
            #tweets.append([float(x) for x in line.split()])
            tweets.append((float(line.split()[0]),
                           float(line.split()[1]),
                           float(line.split()[2])))
    return tweets


# Plots clusters
def plot(clusters):
    map = plt.axes(projection=ccrs.PlateCarree())
    map.stock_img()
    for cluster in clusters:
        colourR = (random.random(), random.random(), random.random())
        map.plot([tweet[1] for tweet in cluster],
                 [tweet[2] for tweet in cluster],
                 "o",
                 color = colourR,
                 markersize = 1,
                 transform = ccrs.Geodetic())
    plt.savefig("map.png")
    #plt.show()


# Computes distance between two tweets.
def tweet_dist (v1, v2):
    #update dmin, dmax
    return math.sqrt((v2[2] - v1[2]) ** 2
                  + ((v2[1] - v1[1]) ** 2))


# Computes bounds on a tweet set.
# Starts from a random tweet, computes the distance to the rest of the
# tweets. Repeats from farthest/shortest tweet until it gets the same
# two tweets. Still quite heuristic... Maybe sufficient. O(n) randomized.
def bound (graph, comp):

    rand_nodes = [graph[i] for i in random.sample(range(len(graph)),3)]

    # Initial bound.
    bnd = tweet_dist(rand_nodes[0], rand_nodes[1])

    # Random starting point.
    v1      = rand_nodes[2]
    v1_prev = [0.0, 0.0, 0.0]
    # Initialise vnext in case we get the bound at first try.
    vnext   = [0.0, 0.0, 0.0]
    v2      = [0.0, 0.0, 0.0]

    while v1 != v1_prev:
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            # Unlucky if bnd starts better than all dists.
            if comp(dist,bnd) and v1 != v2:
                bnd   = dist
                vnext = v2
        v1_prev = v1
        v1      = vnext

    return bnd


# Computes βs.
def betas (dmin, dmax, epsilon):
    n     = (1+epsilon)
    betas = []
    while n < dmax:
        if n > dmin:
            betas.append(n)
        n = n * (1+epsilon)
    return betas


# Builds a cluster within 2β distance of a given center.
def build_cluster (graph, beta, center):
    cluster = set()
    cluster.add(center)
    for node in graph:
        if tweet_dist(center, node) < 2 * beta:
            cluster.add(node)
    return cluster


# Initial cluster.
# Build clusters out of random centers.
def clustering (graph, k, betas):

    result = []
    for beta in betas:
        centers     = set()
        clusters    = []
        unclustered = set(graph)
        nb_centers  = 1

        print(beta, "\t", betas.index(beta)+1, "out of\t", len(betas),
                flush=True)
        while nb_centers < k and len(unclustered):

            #print(nb_centers,"out of\t",k, "unclustered:\t", len(unclustered))
            new_center = unclustered.pop()
            centers.add(new_center)
            clusters.append(build_cluster(unclustered, beta, new_center))

            unclustered = set(x for x in unclustered if x not in clusters[-1])
            nb_centers += 1

        result.append([centers,clusters,unclustered,beta])
    return result


# Returns the index of the the cluster containing the point.
def get_cluster(clusters, point):
    for i in range(len(clusters)):
        if point in clusters[i]:
            return i
    return None


# Insterts a point in an existing cluster if possible, otherwise as a new
# center, otherwise as an unclustered point.
def pointInsertion (L, k, point):
    for i in range(len(L)):
        centers     = L[i][0]
        clusters    = L[i][1]
        unclustered = L[i][2]
        beta        = L[i][3]
        done        = False
        # Inserts in an existing cluster.
        for center in centers:
            dist = tweet_dist(point, center)
            if dist < 2 * beta:
                j = get_cluster(clusters, center)
                clusters[j].add(point)
                done = True
            # Recomputes bounds.
            #if dist > dmax or dist < dmin:
            #    def rebound(dist, L):
            #        global dmax
            #        global dmin
            #        global epsilon
            #        global k
            #        if dist > dmax:
            #            dmax = dist
            #            lbeta = betas(dmin,dmax,epsilon)
            #            if len(L) != len(lbeta):
            #                tmp = set(c for c in clusters)#.union(unclustered)
            #                print(tmp)
            #                L.append(clustering(list(tmp),k,[lbeta[len(lbeta)-len(L):]]))
            #        elif dmin < dist:
            #            dmin = dist
            #            lbeta = betas(dmin,dmax,epsilon)
            #            if len(L) != len(lbeta):
            #                tmp = set(c for c in clusters).union(unclustered)
            #                L.append(clustering(list(tmp)),k,[lbeta[:len(lbeta)-len(L)]])
            #                L.sort(key=lambda l: l[3])
            #    L = rebound(dist,L)
        # Or as a new center.
        if len(centers) < k and not done:
            centers.add(point)
            clusters.append(build_cluster(unclustered, beta, point))
        elif not done:
            unclustered.add(point)
        L[i] = (centers, clusters, unclustered, beta)
    return L


# Deletes a point. If it isn't a center, just deletes it, otherwise reclusters.
def pointDeletion (L, k, point):
    for i in range(len(L)):
        centers     = L[i][0]
        clusters    = L[i][1]
        unclustered = L[i][2]
        beta        = L[i][3]

        if point in unclustered:
            unclustered.remove(point)
        elif point in centers:

            j        = get_cluster(clusters, point)
            new_uncl = set(tweet for cluster in clusters[j:]
                            for tweet in cluster).union(unclustered)
            centers = centers.difference(new_uncl)
            new_uncl.remove(point)
            del clusters [j:]

            tmp      = clustering(new_uncl, k-j+1, [beta])


            centers     = centers.union(tmp[0][0])
            clusters   += tmp[0][1]
            unclustered = unclustered.union(tmp[0][2])
        else:
            for cluster in clusters:
                if point in cluster:
                    cluster.remove(point)

        L[i] = (centers, clusters, unclustered, beta)
    return L


# Returns the smallest β.
def get_solution (L):
    for x in L:
        if not x[2]:
            return x


# Returns the index of the smallest β.
def get_solution_index (L):
    for i in range(len(L)):
        if not L[i][2]:
            return i


# Plots the best solution.
def plot_solution(L):
    plot(L[get_solution_index(L)][1])


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("[*] python3 main.py k epsilon window")
        sys.exit()
    global k
    global epsilon
    global window
    global dmin
    global dmax

    k       = int(sys.argv[1])
    epsilon = float(sys.argv[2])
    window  = int(sys.argv[3])
    if len(sys.argv) > 5:
        stop = int(sys.argv[5])
    else:
        stop = 0

    time  = []
    best_beta = []
    start = timeit.default_timer()
    print("[*] Parsing input file.")
    checkpoint = timeit.default_timer()

    tweets = read_tweets("dataset/twitter_1000000.txt")

    print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    print("[*] Time elapsed:\t", ctime)

    print("[*] Computing dmin.")
    checkpoint = timeit.default_timer()

    dmin   = bound(tweets[:window], operator.lt)

    print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    print("[*] Time elapsed:\t", ctime)
    print("[*] Computing dmax.")
    checkpoint = timeit.default_timer()

    dmax   = bound(tweets[:window], operator.gt)

    print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    print("[*] Time elapsed:\t", ctime)

    print("[*] Got a dmin of:\t", dmin)
    print("[*] Got a dmax of:\t", dmax)

    print("[*] Initial clustering.")
    checkpoint = timeit.default_timer()

    lbetas = betas(dmin,dmax,epsilon)
    L      = clustering(tweets[:window], k, lbetas)

    print("[*] It took:\t\t", timeit.default_timer() - checkpoint)
    ctime = timeit.default_timer() - start
    time.append(ctime)
    print("[*] Time elapsed:\t", ctime)

    best_beta.append(get_solution(L)[3])
    count = window
    if not stop:
        for i in range(len(tweets)-window):
            L = pointInsertion(L, k, tweets[i+(window-1)])
            L = pointDeletion (L, k, tweets[i-(window-1)])
            best_beta.append(get_solution(L)[3])
            if count % 10000 == 0:
                print("[*]", count, "\t\t tweets processed so far.")
                time.append(ctime)
                ctime = timeit.default_timer() - start
                print("[*] Time elapsed:\t", ctime, flush=True)
            count += 1

    print("[*] Best overall β:\t", get_solution(L)[3])
    plot_solution(L)

    print(time)

    print(best_beta)

