import random
import operator
import math


# Read dataset, store tweets as [timestamp, lat, long]
def read_tweets (file):
    tweets=[]
    with open(file) as input:
        for line in input:
            tweet=line.split()
            for i in range(len(tweet)):
                tweet[i] = float(tweet[i])
            tweets.append(tweet)
    return tweets


# Compute distance between two tweets
def tweet_dist(v1, v2):
    return math.sqrt((v2[2] - v1[2]) ** 2
                  + ((v2[1] - v1[1]) ** 2))


# Compute bounds on a tweet set.
# Start from a random tweet, compute the distance to the rest of the
# tweets. Repeat from farthest/shortest tweet until we get the same
# two tweets. Still quite heuristic... O(n) randomized.
def bound(graph, comp):

    i = random.sample(range(len(graph)),3)

    # Init bound
    bnd = tweet_dist(graph[i[0]], graph[i[1]])

    # Random starting point
    v1      = graph[i[2]]
    v1_prev = [0.0, 0.0, 0.0]
    # Init vnext in case we get the bound at first try
    vnext   = [0.0, 0.0, 0.0]
    v2      = [0.0, 0.0, 0.0]

    while v1 != v1_prev:

        #print (v1, v1_prev)
        for v2 in graph:
            dist = tweet_dist(v1,v2)
            # Unlucky if bnd starts better than all dists
            if comp(dist,bnd) and v1 != v2:
                #print(bnd, dist)
                bnd   = dist
                vnext = v2

        v1_prev = v1
        v1      = vnext
    #print(v1,v1_prev)

    return bnd

# Computes the closest pair of points in a tweet set in O(n log n)
# Misses closest pair each on one side of the median
# Quite slow...
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

        # Get the tweets within delta of the border
        border = (graph[median:median+6]
                + graph[median-6:median])

        # Distance to median
        def cmp (v1, v2):
            return (tweet_dist (graph[median], v1)
                  - tweet_dist (graph[median], v2))

    return min(closest_pair_of_points(graph[median:]),
               closest_pair_of_points(graph[:median]))



# Main

tweets=read_tweets("dataset/twitter_1000000.txt")

dmax=bound(tweets, operator.gt)
dmin=bound(tweets, operator.lt)
#for i in range(2):
#    dmax=max(dmax,bound(tweets, operator.gt))

#dmin = closest_pair_of_points(tweets)

print(dmax)
print(dmin)

