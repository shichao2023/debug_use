import numpy as np
import os

def compute(filedir):
	f = open(filedir, 'r')
	lines = f.readlines()
	s = []

	for l in lines:
	    if l != '':
	        s.append(float(l))
	print(np.mean(s[-10:]))	    
# return s
	# print(s / t)
# print(np.me(s[-10:]))
compute('./baseline.txt')
compute('./ours.txt')