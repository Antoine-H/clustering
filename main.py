
#
# Implementation of a 2-approximation algorithm for the fully dynamic k-center
# clustering problem.
# From https://sites.google.com/site/maurosozio/techReport.pdf
#

import random
import operator
import math


# Reads dataset, stores tweets as [timestamp, latitude, longitude].
def read_tweets (file):
    tweets=[]
    with open(file) as input:
        for line in input:
            tweets.append([float(x) for x in line.split()])
    return tweets


# Computes distance between two tweets.
def tweet_dist(v1, v2):
    return math.sqrt((v2[2] - v1[2]) ** 2
                  + ((v2[1] - v1[1]) ** 2))


# Computes bounds on a tweet set.
# Starts from a random tweet, computes the distance to the rest of the
# tweets. Repeats from farthest/shortest tweet until it gets the same
# two tweets. Still quite heuristic... Maybe sufficient. O(n) randomized.
def bound(graph, comp):

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

        #print (v1, v1_prev)
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            # Unlucky if bnd starts better than all dists.
            if comp(dist,bnd) and v1 != v2:
                #print(bnd, dist)
                bnd   = dist
                vnext = v2

        v1_prev = v1
        v1      = vnext
    #print(v1,v1_prev)

    return bnd

# Computes the closest pair of points in a tweet set in O(n log n).
# Misses closest pair each on one side of the median
# From https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
# Quite slow... NOT DONE YET
def closest_pair_of_points (graph):
    if len(graph) == 1:
        # Arbitrary..
        return 100
    if len(graph) == 2:
        return tweet_dist(graph[0], graph[1])
    else:
        sorted_lat  = graph.sort(key=operator.itemgetter(1))
        sorted_long = graph.sort(key=operator.itemgetter(2))

        median      = int(len(graph)/2)

        #delta  = min(closest_pair_of_points(graph[median:]),
        #             closest_pair_of_points(graph[:median]))

        # Get the tweets within delta of the border.
        border = (graph[median:median+6]
                + graph[median-6:median])

        # Distance to median.
        def cmp (v1, v2):
            return (tweet_dist (graph[median], v1)
                  - tweet_dist (graph[median], v2))

    return min(closest_pair_of_points(graph[median:]),
               closest_pair_of_points(graph[:median]))


# Initial cluster
# Can't build cluster, point too far it seems
def clustering (graph, k, dmin, dmax):
    for beta in range(int(dmax-dmin)):
        # Picks k centers
        centers = [graph[i] for i in random.sample(range(len(graph)),k)]
        graph2  = graph[:]
        clusters = []
        for center in centers:
            # Adds all point whose dist < dmax
            #print(center)
            # clusters.append([graph2.pop(graph2.index(x))
            #                 for x in graph2 if tweet_dist(center,x) < beta])
            cluster = []
            for x in graph2:
                if tweet_dist(center, x) < beta:
                    cluster.append(x)
                    graph2.remove(x)
            clusters.append(cluster)
        #if (len(graph) == sum(len(x) for x in clusters)):
        #print(beta)
        #print(len(graph2))
        #print(graph2)
        #print(sum(len(x) for x in clusters))
        if (not len(graph2)):
            return clusters
        if (len(graph) < sum(len(x) for x in clusters)):
            raise AssertionError()
        #print(beta)
        #print("centers")
        #print(centers)
        #print("clusters")
        #print(clusters)
    raise AssertionError()
    return 0

# Main

tweets=read_tweets("dataset/twitter_1000000.txt2")

dmin=bound(tweets, operator.lt)
dmax=bound(tweets, operator.gt)

print(dmin)
print(dmax)

cluster = clustering(tweets, 10, dmin, dmax)
print(cluster)
print(sum(len(x) for x in cluster))


