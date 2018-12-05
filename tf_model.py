
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

		Z_in = tf.sqrt(tf.reduce_sum(self.var_dict['W_in']**2, axis = 0, keep_dims = True))
		Z_rnn = tf.sqrt(tf.reduce_sum(self.var_dict['W_rnn']**2, axis = 0, keep_dims = True))

		self.W_in = self.var_dict['W_in']/Z_in

		for t in range(par['num_time_steps']):

			#input_data = self.input_data[t]/tf.reduce_sum(self.input_data[t]+1e-9, axis=1, keep_dims=True)
			#input_data = self.input_data[t] - tf.reduce_mean(self.input_data[t], axis=(0,1), keep_dims=True)

			h = self.input_data[t] @ self.var_dict['W_in']/Z_in + h @ self.var_dict['W_rnn']/Z_rnn + tf.random_normal(tf.shape(h), 0., 0.05)

			h = tf.nn.softmax(2*tf.nn.sigmoid(c)*h, dim = -1)
			self.raw_hidden_hist.append(h)
			#normalization = 0.01 + h @ par['norm_matrix']
			#h = tf.nn.relu(h - normalization)
			#h /= normalization



			c = (1-par['alpha_neuron'])*c + par['alpha_neuron']*(h @ self.var_dict['W_rnn'])

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
				tf.reduce_mean(tf.abs(self.var_dict['W_in'][:,ind])**1) + \
				tf.reduce_mean(tf.abs(self.var_dict['W_rnn'][:,ind])**1)/10 )


			losses.append(this_loss)
			ops.append(opt.compute_gradients(this_loss))

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
			_, inputs, outputs, mask, _, sample, test, match = stim.generate_trial(task)

			feed_dict = {x:inputs, y:outputs, m:mask}
			_, loss, h, c, hc, W_in = sess.run([model.train, model.loss, model.raw_hidden_hist, model.gate_hist, \
				model.hidden_hist, model.W_in], feed_dict=feed_dict)

			if i%50 == 0:
				print('Iter: {:>4} | h-Loss: {:5.8f} |'.format(i, loss))

				ind = np.zeros((4), dtype = np.int8)
				ind[0] = np.where((test==0)*(match==1))[0][0]
				ind[1] = np.where((test==0)*(match==0))[0][0]
				ind[2] = np.where((test==4)*(match==1))[0][0]
				ind[3] = np.where((test==4)*(match==0))[0][0]

				max_vals = [[] for _ in range(4)]
				min_vals = [[] for _ in range(4)]
				for b in range(4):
					max_vals[0].append(np.max(inputs[:,ind[b],:]))
					max_vals[1].append(np.max(h[:,ind[b],:]))
					max_vals[2].append(np.max(c[:,ind[b],:]))
					max_vals[3].append(np.max(hc[:,ind[b],:]))
					min_vals[0].append(np.min(inputs[:,ind[b],:]))
					min_vals[1].append(np.min(h[:,ind[b],:]))
					min_vals[2].append(np.min(c[:,ind[b],:]))
					min_vals[3].append(np.min(hc[:,ind[b],:]))

				fig, ax = plt.subplots(5,4, figsize=(10,10))
				plt.suptitle('Network States (TF version)')
				for b in range(4):
					x0 = ax[0,b].imshow(inputs[:,ind[b],:].T, aspect='auto', clim = (np.min(min_vals[0]), np.max(max_vals[0])))
					x1 = ax[1,b].imshow(h[:,ind[b],:].T, aspect='auto', clim = (np.min(min_vals[1]), np.max(max_vals[1])))
					x2 = ax[2,b].imshow(c[:,ind[b],:].T, aspect='auto', clim = (np.min(min_vals[2]), np.max(max_vals[2])))
					x3 = ax[3,b].imshow(hc[:,ind[b],:].T, aspect='auto', clim = (np.min(min_vals[3]), np.max(max_vals[3])))
					x4 = ax[4,b].imshow(W_in, aspect='auto')

				ax[0,0].set_ylabel('Inputs')
				ax[1,0].set_ylabel('$\\sigma$(c)')
				ax[2,0].set_ylabel('h')
				ax[3,0].set_ylabel('h*$\\sigma$(c)')
				ax[3,0].set_xlabel('Time Steps')

				ax[0,0].set_title('MATCH Test=0')
				ax[0,1].set_title('NON-MATCH Test=0')
				ax[0,2].set_title('MATCH Test=4')
				ax[0,3].set_title('NON-MATCH Test=4')

				plt.colorbar(x0, ax=ax[0,b])
				plt.colorbar(x1, ax=ax[1,b])
				plt.colorbar(x2, ax=ax[2,b])
				plt.colorbar(x3, ax=ax[3,b])
				plt.colorbar(x4, ax=ax[4,b])
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
