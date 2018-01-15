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


# Compute bounds on set.
# Start from a random tweet, compute the distance to the rest of the
# tweets. Repeat from farthest/shortest tweet until we get the same
# two tweets. Still quite heuristic...
def bound(graph, comp):

    i = random.sample(range(len(graph)),3)

    # Init bound
    bnd = math.sqrt((graph[i[0]][2]-graph[i[1]][2]) ** 2
                  + (graph[i[0]][1]-graph[i[1]][1]) ** 2)

    # Random starting point
    v1      = graph[i[2]]
    v1_prev = [0.0, 0.0, 0.0]
    # Init vnext if we get the bound at first try
    vnext   = [0.0, 0.0, 0.0]
    v2      = [0.0, 0.0, 0.0]
    
    while v1 != v1_prev:
    
        print (v1, v1_prev)
        for v2 in graph:
            dist = math.sqrt((v2[2]-v1[2]) ** 2
                           + (v2[1]-v1[1]) ** 2)
            # Unlucky if bnd starts better than all dists
            if comp(dist,bnd) and v1 != v2:
                print(bnd, dist)
                bnd   = dist
                vnext = v2
    
        v1_prev = v1
        v1      = vnext
    print(v1,v1_prev)

    return bnd


# Main

tweets=read_tweets("dataset/twitter_1000000.txt")

dmax=[]
for i in range(5):
    dmax=max(dmax,bound(tweets, operator.gt))

dmin = bound(tweets, operator.lt)

print(dmax)
# min gets stuck in local optimums
print(dmin)

