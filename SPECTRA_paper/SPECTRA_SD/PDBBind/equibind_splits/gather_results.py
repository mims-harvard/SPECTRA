import os
import pickle

lambda_params = []
props = []
sample_num = []

for filename in os.listdir('.'):
	if '.txt' in filename:
		lambda_param, prop, sample = open(filename, 'r').readlines()[0].split('\t')
		lambda_params.append(float(lambda_param))
		props.append(float(prop))
		sample_num.append(int(sample))

print(lambda_params)
print(props)
print(sample_num)

pickle.dump(lambda_params, open('lambda_param', 'wb'))
pickle.dump(props, open('calculated_proportions', 'wb'))
pickle.dump(sample_num, open('number_samples', 'wb'))
