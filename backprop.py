from utils import *

class BackpropRNN:

	def __init__(self):
		pass

	def calculate_gradients(self, var_dict, hidden, output_grad):
		W_rnn = var_dict['W_rnn']
		W_out = var_dict['W_out']
		b_out = var_dict['b_out']

		for k in range(par['num_time_steps']):
			for j in range(0,k):
				