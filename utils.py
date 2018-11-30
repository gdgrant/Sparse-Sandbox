import sys, time, pickle
import itertools

from parameters import *
import stimulus

import numpy as np
if len(sys.argv) > 1:
	import cupy as cp
	cp.cuda.Device(sys.argv[1]).use()
else:
	import numpy as cp

### GPU utilities

def to_gpu(x):
	""" Move numpy arrays (or dicts of arrays) to GPU """
	if type(x) == dict:
		return {k:cp.asarray(a) for (k, a) in x.items()}
	else:
		return cp.asarray(x)

def to_cpu(x):
	""" Move cupy arrays (or dicts of arrays) to CPU """
	if len(sys.argv) > 1:
		if type(x) == dict:
			return {k:cp.asnumpy(a) for (k, a) in x.items()}
		else:
			return cp.asnumpy(x)
	else:
		if type(x) == dict:
			return {k:a for (k, a) in x.items()}
		else:
			return x

def relu(x):
	return np.maximum(0., x)

def sigmoid(x):
	return cp.exp(x)/(cp.exp(x)+1)

def softmax(x, a=-1):
	c = cp.exp(x-cp.amax(x, axis=a, keepdims=True))
	return c/cp.sum(c, axis=a, keepdims=True)

def accuracy(output, target, mask, inc_fix=False):
	""" Calculate accuracy from output, target, and mask for the networks """
	output = output
	target = target
	mask   = mask

	arg_output = cp.argmax(output, -1)
	arg_target = cp.argmax(target, -1)
	mask = mask if inc_fix else mask * (arg_target != par['n_output']-1)

	return cp.sum(mask * (arg_output == arg_target), axis=(0,1))/cp.sum(mask, axis=(0,1))

