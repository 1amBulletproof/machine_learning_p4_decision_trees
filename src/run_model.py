#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	run experiment

from classification_tree import ClassificationTree

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
import numpy as np

#=============================
# run_model()
#
#	- read-in 5 groups of input data, train on 4/5,
#		test on 5th, cycle the 4/5 & repeat 5 times
#		Record overall result!
#=============================
def run_model_with_cross_validation(prune=False):

	#GET DATA
	#- expect data_0 ... data_4
	data_groups = list()
	data_groups.append(FileManager.get_csv_file_data_array('data_0'))
	data_groups.append(FileManager.get_csv_file_data_array('data_1'))
	data_groups.append(FileManager.get_csv_file_data_array('data_2'))
	data_groups.append(FileManager.get_csv_file_data_array('data_3'))
	data_groups.append(FileManager.get_csv_file_data_array('data_4'))
	
	if prune == True:
		validation_data = FileManager.get_csv_file_data_array('validation_data')

	NUM_GROUPS = len(data_groups)

	#For each data_group, train on all others and test on me
	culminating_result = 0;
	culminating_validation_result = 0;

	finanl_average_result = 0
	final_validation_average_result = 0

	for test_group_id in range(NUM_GROUPS):
		print()
		#Form training data as 4/5 data
		train_data = list()
		for train_group_id in range(len(data_groups)):
			if (train_group_id != test_group_id):
				#Initialize train_data if necessary
				if (len(train_data) == 0):
					train_data = data_groups[train_group_id]
				else:
					train_data = train_data + data_groups[train_group_id]

		print('train_data group', str(test_group_id), 'length: ', len(train_data))
		#print(train_data)

		test_data = data_groups[test_group_id]

		result = 0
		validation_result = 0
		model = ClassificationTree(train_data)
		model.train()
		print('tree size:', model.get_size_of_tree())
		print('tree: ')
		print(model.print_tree())
		result= model.test(test_data)
		#print('result:', result)
		culminating_result= culminating_result + result
		print('Accuracy (%):', result)
		if prune == True:
			model.validate(validation_data)
			print('tree size w/ pruning:', model.get_size_of_tree())
			print('tree: ')
			print(model.print_tree())
			validation_result= model.test(test_data)
			#print('result:', result)
			culminating_validation_result= culminating_validation_result + validation_result
			print('Accuracy w/ pruning (%):', validation_result)
		print()


	final_average_result = culminating_result / NUM_GROUPS
	final_validation_average_result = culminating_validation_result / NUM_GROUPS
	#print()
	#print('final average result:')
	#print(final_average_result)
	#print()

	return (final_average_result, final_validation_average_result)


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Run the classification tree test')
	parser.add_argument('prune', type=str, help='do pruning? accepts "yes":"no" or "true":"false", expects "validation_data" file if-so')
	args = parser.parse_args()
	print(args)
	prune_str = args.prune
	prune_str = prune_str.lower() 

	prune = False
	if prune_str == "true" or prune_str == "yes":
		prune = True

	final_result = run_model_with_cross_validation(prune)
	print('Average Accuracy (%):') 
	print(final_result[0], '%')
	if (prune):
		print('Average pruned Accuracy (%):')
		print(final_result[1], '%')


if __name__ == '__main__':
	main()
