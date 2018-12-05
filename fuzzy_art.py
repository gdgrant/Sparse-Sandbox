from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model:

	def __init__(self):
		self.load_weights()

		# ART parameters
		self.a   = 0.001	# choice
		self.b   = 0.1		# learning rate
		self.rho = 0.1		# vigilance

	def load_weights(self):
		self.var_dict = {}
		self.var_dict['W_in'] = to_gpu(np.ones_like(par['W_in_init']))

	def load_batch(self, input_data, output_data, mask):
		self.input_data = to_gpu(input_data/input_data.max())
		self.output_data = to_gpu(output_data)
		self.mask = to_gpu(mask)

	def run_model(self):
		
		# Input:  I = [time,batch,M]	Categories 'i', from 1 to M
		# Weight: W = [M,N]				
		# Hidden: T = [time,batch,N]	Categories 'j', from 1 to N

		self.h_record = cp.zeros([par['num_time_steps'],par['batch_size'],par['n_hidden']])
		self.resonance = cp.zeros([par['num_time_steps'],par['batch_size']])

		for t in range(par['num_time_steps']):

			fuzzy_and = cp.minimum(self.input_data[t,...,cp.newaxis], self.var_dict['W_in'])
			T_state = cp.sum(fuzzy_and, axis=1)/(self.a + cp.sum(self.var_dict['W_in'], axis=0))
			T_choice = cp.argsort(T_state, axis=1)[::-1]

			for n in range(par['n_hidden']):
				print(fuzzy_and.shape)
				quit()
				vigilance = cp.sum(fuzzy_and[:,T_choice[]])







			self.h_record[t] = T

	def optimize(self):
		pass



def main():

	stim  = stimulus.MultiStimulus()
	model = Model()

	task = 0
	for i in range(par['n_train_batches']):
		_, inputs, outputs, mask, _ = stim.generate_trial(task)
		model.load_batch(inputs, outputs, mask)
		model.run_model()
		model.optimize()

		if i%25 == 0:

			fig, ax = plt.subplots(2,4,figsize=(10,10))
			plt.suptitle('Fuzzy ART Input and Hidden State')
			for b in range(4):
				x0 = ax[0,b].imshow(inputs[:,b,:].T, aspect='auto')
				x1 = ax[1,b].imshow(to_cpu(model.h_record[:,b,:].T), aspect='auto')

			ax[0,0].set_ylabel('Inputs')
			ax[1,0].set_ylabel('Hidden State')

			plt.colorbar(x0, ax=ax[0,b])
			plt.colorbar(x1, ax=ax[1,b])
			plt.savefig('./plotdir/iter{}.png'.format(i))
			plt.clf()
			plt.close()

			print('Iter: {:>4}'.format(i))


main()