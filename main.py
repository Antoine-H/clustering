with open("../dataset/twitter_1000000.txt") as input:
        print zip(*(line.strip().split('\t') for line in input))

