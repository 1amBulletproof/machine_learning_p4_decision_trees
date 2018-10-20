#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	General Data Manipulation 

import argparse
import copy
import random
import numpy as np
import pandas as pd


#=============================
# DATAMANIPULATOR
#
#	- Contains methods to useful for pre-processing data (typically 2D matrixes)
#@TODO: replace some funcitonality with support for numpy / pandas
#=============================
class DataManipulator:

	#=============================
	# EXPAND_ATTRIBUTE_TO_BINARY_VALUES()
	#
	#	- Expand attribute col to multiple cols for given multi-bin valued attribute
	#	- Creates copy of input data - input data unmodified but slower
	#	- This is sometimes referred as "one-hot coding"
	#
	#@param	data		2D matrix
	#@param	col_idx		column attribute to expand
	#@param	num_bins	max number of bins 
	#@return			modified 2D matrix with ADDITIONAL cols for each bin 
	#=============================
	@staticmethod
	def expand_attributes_to_binary_values(data, col_idx, num_bins):
		modified_data = list(data)
		modified_data = copy.deepcopy(data)
		length_of_vector = len(data[0])

		if col_idx >= length_of_vector:
			print('ERROR: Column idx ' , col_idx , ' outside input data cols ' , length_of_vector)
			return 

		row_idx = 0
		for row in modified_data:
			row_val = row[col_idx]
			binary_category_values = DataManipulator._convert_bin_val_into_binary_vector(row_val, num_bins)

			insert_idx = col_idx
			#replace previous value
			modified_data[row_idx][insert_idx:(insert_idx+1)] = binary_category_values

			row_idx += 1

		return modified_data

	#=============================
	# _CONVERT_BIN_VAL_INTO_BINARY_VECTOR()
	#
	#	- Private fcn (helper method)
	#	- creates vector of binary values to represent one hot coding
	#	- 0 - base (Assumes 0 is a valid value)
	#	- i.e. val 3 && num_bins 5 -> [0,0,0,1,0]
	#=============================
	@staticmethod
	def _convert_bin_val_into_binary_vector(val, num_bins):
		bin_vals = [0 for val in range(num_bins)]
		bin_vals[int(val)] = 1
		return bin_vals

	#=============================
	# MOVE_NP_COLUMN_TO_END()
	#
	#	- mv given column to be the last column in numpy 2D matrix
	#	- returns a new numpy 2D array (i.e. creates one from scratch, doesn't modify input)
	#
	#@param data		numpy 2D numpy data array
	#@param col	column to move to the last column
	#@return			modified data as numpy matrix
	#=============================
	@staticmethod
	def move_np_column_to_end(data, col):
		#Get the num cols as list 0...len(col)
		col_list = list(range(0,data.shape[1]))
		#REMOVE 'col' from the list
		col_list.remove(col)
		#Append 'col' to the end of the list
		col_list.append(col)
		numpy_result = data[:,col_list]
		return numpy_result

	#=============================
	# MOVE_COLUMN_TO_END()
	#
	#	- mv given column to be the last column in 2D matrix
	#	- returns a new 2D array (i.e. creates one from scratch, doesn't modify input)
	#
	#@param data		2D data array
	#@param col_idx	column to move to the last column
	#@return			modified data
	#=============================
	@staticmethod
	def move_column_to_end(data, col):
		#print('LOG: move_column_to_end() START')
		modified_data = []
		length_of_vector = len(data[0])

		if col >= length_of_vector:
			print('ERROR: Column idx ' , col , ' outside input data cols ' , length_of_vector)
			return 

		row_idx = 0
		for row in data:
			modified_data.append([])
			col_idx = 0
			swap_val = 'BLAH'
			for val in row:
				if col == col_idx:
					swap_val = val
				else:
					modified_data[row_idx].append(val)
				col_idx += 1

			modified_data[row_idx].append(swap_val)
			row_idx += 1
		
		#print(input_data)
		#print('LOG: move_column_to_end() END')
		return modified_data

	#=============================
	# SPLIT_DATA_IN_2_RANDOMLY()
	#
	#	- Split given input data (expected 2D array) into 2 pieces (one is typically for learning, the other testing) based on given fraction
	#
	#@param data		expected 2d matrix to split up
	#@param fraction	fraction of split in the first group
	#@return			tuple(data_slice1_as_2d_matrix, data_slice2_as_2d_matrix)
	#=============================
	@staticmethod
	def split_data_in_2_randomly(data, fraction):

		#print('data before shuffle:')
		#print(data)

		#Randomize the data
		data_copy = copy.deepcopy(data)
		random.shuffle(data_copy)

		#print('data after shuffle:')
		#print(data)
		
		lines = round((10 * fraction), 1)
		if lines < 1 or lines > 9:
			print('ERROR: bad fraction (too small or large) ' , fraction)
			return

		data_set_1 = list()
		data_set_2 = list()
		row_counter = 0
		for row in data_copy:
			if row_counter <= lines:
				data_set_1.append(row)
			elif row_counter > lines:
				data_set_2.append(row)

			if row_counter == 10:
				row_counter = 0
			row_counter += 1

		return (data_set_1, data_set_2)

	#=============================
	# SPLIT_DATA_RANDOMLY
	#
	#	- Split given input data randomly into groups
	#	- Assumes data is dataframe
	#
	#@param data		input data as data frame
	#@param number_of_splits	number of groups returned
	#@return			list numpy 2d arrays (data_set1, ... data_set_number_of_splits)
	#=============================
	@staticmethod
	def split_data_randomly(data, number_of_splits=5):
		#Randomize the data
		data_copy = copy.deepcopy(data.values)
		np.random.shuffle(data_copy)
		num_rows = len(data_copy)

		#divy up slices based purely on length of data
		# i.e. 21 samples, 5 groups, each group gets 4 samples (final group gets 5
		rows_in_data = len(data_copy)
		rows_per_group = int(rows_in_data/number_of_splits)
		if rows_per_group < 1:
			print('not enough data', rows_in_data, ' for ', number_of_splits, 'splits')
			return

		numpy_groups = list()
		prev_cuttoff = 0
		for group_idx in range(number_of_splits):
			numpy_groups.append(
					data_copy[prev_cuttoff:prev_cuttoff+rows_per_group])
			prev_cuttoff = prev_cuttoff + rows_per_group 

		return numpy_groups

	#=============================
	# SPLIT_DATA_RANDOMLY_ACCOUNTING_FOR_CLASS
	#
	#	- Split given input data randomly into groups
	#		- Each group will have approx.same number of a given class
	#		- Assumes classification is final column!
	#		- Assumes data is dataframe
	#
	#@param data		input data as data frame
	#@param number_of_splits	number of groups returned
	#@return			list numpy 2d arrays (data_set1, ... data_set_number_of_splits)
	#=============================
	@staticmethod
	def split_data_randomly_accounting_for_class(data, number_of_splits=5):
		'''
		idea here is simple:

		1. sort the data by classification (final col.) so it's like so:

			1.   1, 5, 1, A
			2.   3, 5, 2, A
			3.   3, 4, 2, A
			4.   6, 6, 6, B
			......

		2. Knowing the number of splits, create a list of row numbers which belong to each group but do so in a random order every time
			group_ids = [1, 2, 3]
			groups = [ 1[], 2[], 3[] ]
			randomize the order & add values to groups
				group order 1 = [2, 1, 3]

				groups[2].append(row_1)
				groups[1].append(row_2)
				groups[3].append(row_3)

			randomize the order & add values to groups
				group order 2 = [1, 3, 2]
				groups[1].append(row_4)
				groups[3].append(row_5)
				groups[2].append(row_6)

				......
		'''
			
		#Sort the data by class and transform to numpy 2darray
		#print('og data')
		#print(data)
		cols = data.columns
		#print('cols')
		#print(cols)
		final_col = data.columns.values[-1]
		#print('final_col')
		#print(final_col)
		data_copy = data.sort_values(by=[data.columns.values[-1]]).values
		#print('sorted data ')
		#print(data_copy)

		#make array of range 1 - number_of_splits
		group_ids = list(range(0, number_of_splits))
		groups = list()
		#Initialize groups
		for idx in range(len(group_ids)):
			groups.append(list())

		random.shuffle(group_ids)
		group_idx = 0
		next_class = -1
		for row_idx in range(len(data_copy)):
			prev_class = data_copy[row_idx][-1]
			group_id = group_ids[group_idx] # get the next group number id
			groups[group_id].append(row_idx)
			if ( row_idx < (len(data_copy) - 2) ):
				next_class = data_copy[row_idx+1][-1]
			group_idx = group_idx + 1

			if (prev_class != next_class):
				#shuffle the order of adding data to groups
				np.random.shuffle(group_ids)
				#Start from the beginning adding features
				group_idx = 0
			elif ( group_idx == number_of_splits):
				#Start from the beginning adding features
				group_idx = 0

		numpy_groups = list()
		for group in groups:
			numpy_groups.append(data_copy[group])

		return numpy_groups

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to test class "DataManipulator"')
	parser = argparse.ArgumentParser(description='Manipulate Data')
	#parser.add_argument('file_path', metavar='F', type=str, help='full path to input file')
	#args = parser.parse_args()

	print()
	print('TEST 1: move class to end')
	test_data = [['classA', 0, 1, 2], ['classB', 2, 1, 0]]
	print('test_data:')
	print(test_data)

	col_to_move_to_end = 0
	moved_data = DataManipulator.move_column_to_end(test_data, col_to_move_to_end)
	print('modified_data:')
	print(moved_data)

	print()
	print('TEST 2: expand attributes')
	print('test_data: ')
	print(moved_data)

	num_bins = 3
	col = 0 
	discrete_data = DataManipulator.expand_attributes_to_binary_values(moved_data, col, num_bins)
	col = 3
	discrete_data = DataManipulator.expand_attributes_to_binary_values(discrete_data, col, num_bins)
	col = 6
	discrete_data = DataManipulator.expand_attributes_to_binary_values(discrete_data, col, num_bins)

	print('modified_data:)')
	print(discrete_data)

	print()
	print('TEST 3: split data')
	test_data.append(['classC', 0, 1, 2])
	test_data.append(['classD', 0, 1, 2])
	test_data.append(['classE', 0, 1, 2])
	test_data.append(['classF', 0, 1, 2])
	test_data.append(['classG', 0, 1, 2])
	test_data.append(['classH', 0, 1, 2])
	test_data.append(['classI', 0, 1, 2])
	test_data.append(['classJ', 0, 1, 2])
	print('test_data: ')
	print(test_data)

	data_sets = DataManipulator.split_data_in_2_randomly(test_data, 0.7)
	print('split_data: ')
	print(data_sets[0])
	print(data_sets[1])

	print()
	print('TEST 4: split data into 5 groups')
	test_data2 = [[1,1,1,'A'],
			[1,1,1,'B'],
			[2,2,2,'B'],
			[3,3,3,'B'],
			[4,4,4,'B'],
			[5,5,5,'B'],
			[2,2,2,'A'],
			[3,3,3,'A'],
			[4,4,4,'A'],
			[5,5,5,'A'],
			[1,1,1,'C'],
			[2,2,2,'C'],
			[3,3,3,'C'],
			[4,4,4,'C'],
			[5,5,5,'C'],
			[1,1,1,'D'],
			[2,2,2,'D'],
			[3,3,3,'D'],
			[4,4,4,'D'],
			[5,5,5,'D'],
			[1,1,1,'E'],
			[2,2,2,'E'],
			[3,3,3,'E'],
			[4,4,4,'E'],
			[5,5,5,'E'],
			[6,6,6,'E']]
	test_dataframe = pd.DataFrame(test_data2)
	print('test_dataframe:')
	print(test_dataframe)
	random_groups = DataManipulator.split_data_randomly(test_dataframe)
	print('random groups:')
	for random_group in random_groups:
		print('len of random group: ', len(random_group))
		print(random_group)
	random_even_groups = DataManipulator.split_data_randomly_accounting_for_class(test_dataframe)
	for random_even_group in random_even_groups:
		print('len of random even group: ', len(random_even_group))
		print(random_even_group)
	

''' COMMENTED OUT FOR SUBMITTAL
if __name__ == '__main__':
	main()
	'''
