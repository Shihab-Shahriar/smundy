import numpy as np
from collections import defaultdict

def intersect(a,b,R):
    min_corner = a-R
    max_corner = a+R
    other_min_corner = b-R
    other_max_corner = b+R
    for d in range(3):
        if min_corner[d] > other_max_corner[d] or max_corner[d] < other_min_corner[d]:
            return False 
    return True


data = None
with open("out.txt") as fh:
    data = fh.readlines()[3:]

N = 1000
R = 4.0

points = []
for i in range(N):
    points.append(list(map(float, data[i].split(','))))

points = np.array(points)

neighs = defaultdict(list)
for x in range(N+3, len(data)):
    i, j = list(map(int, data[x].strip().split(' ')))
    neighs[i].append(j)

for i in neighs:
    neighs[i] = sorted(neighs[i])

for i in range(N):
    nn = []
    for j in range(N):
        if i==j: continue
        #dist = np.power(points[i]-points[j],2).sum()**.5
        if intersect(points[i], points[j], R):
            nn.append(j)
    assert nn==neighs[i], f"{i}, {nn}, {neighs[i]}, {len(nn)}, {len(neighs[i])}"

