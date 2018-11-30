
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parameters import *
import stimulus
from AdamOpt import AdamOpt

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model:

	def __init__(self, input_data, target_data, mask):

		self.input_data = tf.unstack(input_data, axis=0)
		self.target_data = tf.unstack(target_data, axis=0)
		self.mask = mask

		self.declare_variables()
		self.run_model()
		self.optimize()

	def declare_variables(self):
		
		self.var_dict = {}
		
		W_in_list = []
		W_rnn_list = []
		for j in range(par['n_hidden']):
			W_in_list.append(tf.get_variable('W_in{}'.format(j), initializer=par['W_in_init'][:,j]))
			W_rnn_list.append(tf.get_variable('W_rnn{}'.format(j), initializer=par['W_rnn_init'][:,j]))
		self.var_dict['W_in'] = tf.stack(W_in_list, axis=-1)
		self.var_dict['W_rnn'] = tf.stack(W_rnn_list, axis=-1) * tf.constant(par['W_rnn_mask'])

		self.hidden_optimizers = []
		for j in range(par['n_hidden']):
			self.hidden_optimizers.append(AdamOpt([W_in_list[j],W_rnn_list[j]]))

	def run_model(self):

		self.raw_hidden_hist = []
		self.hidden_hist = []
		self.gate_hist = []
		h = tf.zeros([par['batch_size'], par['n_hidden']])
		c = tf.zeros([par['batch_size'], par['n_hidden']])

		for t in range(par['num_time_steps']):
			c = (1-par['alpha_neuron'])*c + par['alpha_neuron']*(h @ self.var_dict['W_rnn'])
			h = self.input_data[t] @ self.var_dict['W_in'] + h @ self.var_dict['W_rnn']
			h = tf.nn.relu(h)
			h = h / tf.reduce_max(h+1e-9, axis=1, keepdims=True)

			self.raw_hidden_hist.append(h)

			h = h * tf.nn.sigmoid(c)

			self.hidden_hist.append(h)
			self.gate_hist.append(tf.nn.sigmoid(c))

		self.raw_hidden_hist = tf.stack(self.raw_hidden_hist, axis=0)
		self.hidden_hist = tf.stack(self.hidden_hist, axis=0)
		self.gate_hist = tf.stack(self.gate_hist, axis=0)


	def optimize(self):
		
		loss = tf.unstack(tf.reduce_mean(self.hidden_hist**2, axis=(0,1)), axis=0)

		ops = []
		losses = []
		for ind, (l, opt) in enumerate(zip(loss, self.hidden_optimizers)):

			this_loss  = -par['spike_cost']*l
			this_loss += par['weight_cost']*( \
				tf.reduce_mean(tf.abs(self.var_dict['W_in'][:,ind])) + \
				tf.reduce_mean(tf.abs(self.var_dict['W_rnn'][:,ind])) )

			losses.append(this_loss)
			ops.append(opt.compute_gradients(l))

		self.train = tf.group(*ops)
		self.loss = tf.reduce_mean(tf.stack(losses))


def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()

	x = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_input']], 'stim')
	y = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size'], par['n_output']], 'out')
	m = tf.placeholder(tf.float32, [par['num_time_steps'], par['batch_size']], 'mask')

	stim = stimulus.MultiStimulus()

	# gpu_options = tf.GPUOptions()
	# gpu_options.per_process_gpu_memory_fraction = 0.99
	# gpu_options.allow_growth = True

	# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	with tf.Session() as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, y, m)

		sess.run(tf.global_variables_initializer())

		print('--> Model successfully initialized.\n')

		task = 1
		for i in range(par['n_train_batches']):
			_, inputs, outputs, mask, _ = stim.generate_trial(task)
			
			feed_dict = {x:inputs, y:outputs, m:mask}
			_, loss, h, c, hc = sess.run([model.train, model.loss, model.raw_hidden_hist, model.gate_hist, model.hidden_hist], feed_dict=feed_dict)

			if i%25 == 0:
				print('Iter: {:>4} | h-Loss: {:5.3f} |'.format(i, loss))

				fig, ax = plt.subplots(4,4, figsize=(10,10))
				plt.suptitle('Network States (TF version)')
				for b in range(4):
					x0 = ax[0,b].imshow(inputs[:,b,:].T, clim=(0,par['tuning_height']), aspect='auto')
					x1 = ax[1,b].imshow(h[:,b,:].T, aspect='auto')
					x2 = ax[2,b].imshow(c[:,b,:].T, aspect='auto')
					x3 = ax[3,b].imshow(hc[:,b,:].T, aspect='auto')

				ax[0,0].set_ylabel('Inputs')
				ax[1,0].set_ylabel('$\\sigma$(c)')
				ax[2,0].set_ylabel('h')
				ax[3,0].set_ylabel('h*$\\sigma$(c)')
				ax[3,0].set_xlabel('Time Steps')

				plt.colorbar(x0, ax=ax[0,b])
				plt.colorbar(x1, ax=ax[1,b])
				plt.colorbar(x2, ax=ax[2,b])
				plt.colorbar(x3, ax=ax[3,b])
				plt.savefig('./plotdir/iter{}.png'.format(i))
				plt.clf()
				plt.close()


if __name__ == '__main__':
    try:
        if len(sys.argv) > 1:
            main(sys.argv[1])
        else:
            main()
    except KeyboardInterrupt:
        print('Quit by KeyboardInterrupt.')