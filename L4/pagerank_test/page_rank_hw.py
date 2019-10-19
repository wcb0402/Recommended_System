import numpy as np
'''

'''
data_x = np.array([[0,1/4,0,1/4,1/4,1/4],[0,0,1,0,0,0],[0,0,0,0,1,0],[1/3,0,1/3,0,1/3,0],[0,1/2,1/2,0,0,0],[0,0,0,1,0,0]]).T
print(data_x)
w0 = np.array([1/6 for i in range(6)])
d= 0.85
n = 6
def simple_pagerank(d,n,w0):
    for i in range(100):
        w0 = (1-d)/n + d*np.dot(data_x,w0)
    print(w0)

def random_pagerankL(w0):
    for i in range(100):
        w0 = np.dot(data_x,w0)
    print(w0)

simple_pagerank(d,n,w0)
random_pagerankL(w0)