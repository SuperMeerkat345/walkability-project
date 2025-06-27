# quick script to count how much data i actually have
import os

def count_num_tracts():
    print(len(os.listdir("./ClevelandData")))

count_num_tracts()
