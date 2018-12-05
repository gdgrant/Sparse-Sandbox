from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


### Building model

class Model:

	def __init__(self):
		self.load_weights()

	def load_weights(self):
		self.var_dict = {}
		self.var_dict['W_in'] = to_gpu(par['W_in_init'])
		self.var_dict['W_out'] = to_gpu(par['W_out_init'])
		self.var_dict['W_rnn'] = to_gpu(par['W_rnn_init'])
		self.var_dict['b_out'] = to_gpu(par['b_out_init'])

	def load_batch(self, input_data, output_data, mask):
		self.input_data = to_gpu(input_data)
		self.output_data = to_gpu(output_data)
		self.mask = to_gpu(mask)

	def run_model(self):

		self.h_record = cp.zeros([par['num_time_steps'],par['batch_size'],par['n_hidden']])
		self.c_record = cp.zeros([par['num_time_steps'],par['batch_size'],par['n_hidden']])
		self.c = cp.zeros([par['batch_size'],par['n_hidden']])
		self.y = cp.zeros([par['num_time_steps'],par['batch_size'],par['n_output']])

		for t in range(par['num_time_steps']):
			recur = cp.matmul(self.h_record[t-1], self.var_dict['W_rnn'])
			noise = cp.random.normal(0,0.01,size=recur.shape)
			state = relu(cp.matmul(self.input_data[t], self.var_dict['W_in']) + recur + noise)
			state *= sigmoid(self.c)

			self.c_record[t] = sigmoid(self.c)
			self.h_record[t] = state/cp.sum(state+1e-9, axis=1, keepdims=True)
			self.c = (1-par['alpha_neuron'])*self.c + par['alpha_neuron']*cp.matmul(self.h_record[t], self.var_dict['W_rnn'])

			self.y[t] = softmax(cp.matmul(self.h_record[t], self.var_dict['W_out'])+self.var_dict['b_out'])

	def optimize(self):

		self.loss = cp.mean(-cp.sum(self.output_data*cp.log(self.y), axis=-1))
		self.accuracy = accuracy(self.y, self.output_data, self.mask)
		
		for i in range(par['n_hidden']):
			delta = cp.mean(self.h_record[...,i:i+1]*self.input_data, axis=(0,1))
			self.var_dict['W_in'][:,i] += delta
		for j in range(par['n_input']):
			self.var_dict['W_in'][j,:] = (self.var_dict['W_in'][j,:]/self.var_dict['W_in'][j,:].max())**2

		error = self.y - self.output_data
		self.var_dict['b_out'] -= 0.1*np.mean(error, axis=(0,1))
		for j in range(par['n_hidden']):
			delta = np.mean(error*self.h_record[:,:,j:j+1], axis=(0,1))
			self.var_dict['W_out'][j,:] -= delta

		"""
		for i in range(par['n_hidden']):
			delta = 0.
			for t in range(1,par['num_time_steps']):
				delta += par['alpha_neuron']**t*cp.mean(self.h_record[t:,:,i:i+1]*self.h_record[:-t,:,:], axis=(0,1))
			#delta = par['alpha_neuron']**1*cp.mean(self.h_record[1:,:,i:i+1]*self.h_record[:-1,:,:], axis=(0,1)) \
			#	  + par['alpha_neuron']**2*cp.mean(self.h_record[2:,:,i:i+1]*self.h_record[:-2,:,:], axis=(0,1)) \
			#	  + par['alpha_neuron']**3*cp.mean(self.h_record[3:,:,i:i+1]*self.h_record[:-3,:,:], axis=(0,1))
		"""	
		delta = 0.
		for t in range(1,par['num_time_steps']):
			a = par['alpha_neuron']*(1-par['alpha_neuron'])**(t-1)
			delta += a*cp.mean(self.h_record[t:,:,:,cp.newaxis]*self.h_record[:-t,:,cp.newaxis,:], axis=(0,1))
		self.var_dict['W_rnn'] += delta
		for j in range(par['n_hidden']):
			self.var_dict['W_rnn'][j,:] = (self.var_dict['W_rnn'][j,:]/self.var_dict['W_rnn'][j,:].max())**2
		for k in range(par['n_hidden']):
			self.var_dict['W_rnn'][k,k] = 0.


def main():

	stim  = stimulus.MultiStimulus()
	model = Model()

	for i in range(par['n_train_batches']):
		task = 16

		_, inputs, outputs, mask, _ = stim.generate_trial(task)
		model.load_batch(inputs, outputs, mask)
		model.run_model()
		model.optimize()

		if i%25 == 0:
			print('Iter: {:>4} | Loss: {:5.3f} | Resp. Acc: {:5.3f} |'.format(i, to_cpu(model.loss), to_cpu(model.accuracy)))

			fig, ax = plt.subplots(4,4, figsize=(10,10))
			plt.suptitle('Input, Hidden State, and Output of Network')
			for b in range(4):
				x0 = ax[0,b].imshow(inputs[:,b,:].T, clim=(0,par['tuning_height']), aspect='auto')
				x1 = ax[1,b].imshow(to_cpu(model.c_record[:,b,:].T), aspect='auto')
				x2 = ax[2,b].imshow(to_cpu(model.h_record[:,b,:].T), aspect='auto', clim=(0,1))
				x3 = ax[3,b].plot(np.argmax(to_cpu(model.y[:,b,:]), axis=-1))

			ax[0,0].set_ylabel('Inputs')
			ax[1,0].set_ylabel('$\\sigma$(c)')
			ax[2,0].set_ylabel('h*$\\sigma$(c)')
			ax[3,0].set_ylabel('Outputs')
			ax[3,0].set_xlabel('Time Steps')

			plt.colorbar(x0, ax=ax[0,b])
			plt.colorbar(x1, ax=ax[1,b])
			plt.colorbar(x2, ax=ax[2,b])
			#plt.colorbar(x3, ax=ax[3,b])
			plt.savefig('./plotdir/iter{}.png'.format(i))
			plt.clf()
			plt.close()

main()