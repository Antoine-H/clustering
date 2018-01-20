
#
# Implementation of a 2-approximation algorithm for the fully dynamic k-center
# clustering problem.
# From https://sites.google.com/site/maurosozio/techReport.pdf
#
# TODO : closest_pair_of_points
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
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            # Unlucky if bnd starts better than all dists.
            if comp(dist,bnd) and v1 != v2:
                bnd   = dist
                vnext = v2
        v1_prev = v1
        v1      = vnext

    return bnd


# Computes the closest pair of points in a tweet set in O(n log n).
# Misses closest pair each on one side of the median.
# From https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
# Quite slow... NOT DONE YET, point on each side of the border.
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
    cluster = [center]
    for node in graph:
        if tweet_dist(center, node) < 2 * beta:
            cluster.append(node)
    return cluster


# Initial cluster.
# Build clusters out of random centers.
def clustering (graph, k, betas):

    result = []
    for beta in betas:
        centers     = []
        clusters    = []
        unclustered = graph[:]
        nb_centers  = 1

        while nb_centers < k and len(unclustered):

            centers.append(random.choice(unclustered))
            unclustered.remove(centers[-1])
            clusters.append(build_cluster(unclustered, beta, centers[-1]))

            unclustered = [x for x in unclustered if x not in clusters[-1]]
            nb_centers += 1

        result.append([centers,clusters,unclustered,beta])
    return result


# Insterts a point in an existing cluster if possible, otherwise as a new
# center, otherwise as an unclustered point.
def pointInsertion(L, k, point):
    for i in range(len(L)):
        centers     = L[i][0]
        clusters    = L[i][1]
        unclustered = L[i][2]
        beta        = L[i][3]
        done        = False

        # Inserts in an existing cluster.
        for center in centers:
            if tweet_dist (point, center) < 2 * beta:
                i = centers.index(center)
                clusters[i].append(point)
                done = True
        # Inserts as a new center.
        if len(centers) < k and not done:
            centers.append(point)
            clusters.append(build_cluster(unclustered, beta, centers[-1]))
        elif not done:
            unclustered.append(point)
    return L


# Flattens a list of lists into a list
def flatten (lls): return [l for ls in lls for l in ls]

# Deletes a point. If it isn't a center, just deletes it, otherwise reclusters.
def pointDeletion(L, k, point):
    for i in range(len(L)):
        centers     = L[i][0]
        clusters    = L[i][1]
        unclustered = L[i][2]
        beta        = L[i][3]

        if point in flatten(clusters) and point not in centers:
            for cluster in clusters:
                if point in cluster:
                    cluster.remove(point)
        elif point in unclustered:
            unclustered.remove(point)
        elif point in centers:

            j        = centers.index(point)
            new_uncl = flatten(clusters[j:]) + unclustered
            new_uncl.remove(point)
            tmp      = clustering (new_uncl, k-j+1, [beta])

            del centers  [j:]
            del clusters [j:]

            centers    += tmp[0][0]
            clusters   += tmp[0][1]
            unclustered = tmp[0][2]
    return L


def get_solution (L):
    for x in L:
        if not x[2]:
            return x


# Main

tweets = read_tweets("dataset/twitter_1000000.txt")

#dmin  = closest_pair_of_points(tweets)
dmin   = bound(tweets, operator.lt)
dmax   = bound(tweets, operator.gt)

L      = clustering(tweets, 4, betas(dmin, dmax, 0.5))

print(L)

L = pointDeletion(L,4,[1504866209.0, 12.3267, 45.4386])

