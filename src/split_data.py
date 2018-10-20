#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			10/30/2018
#@description	split data in number of groups


import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
import numpy as np
import csv

#=============================
# split_data_90_10
#
#	- read-in data and split it into 5 pieces
#=============================
def split_data(file_path, is_random, num_groups=5, separator=','):

	#GET DATA
	original_data = FileManager.get_csv_file_data_pandas(file_path, separator)
	#print('original data')
	#print(original_data)

	#STANDARD STUFF

	#Split the data into 5 groups
	groups = list()
	num_groups = int(num_groups)
	
	if (is_random == True):
		#Use basic random 5-way split
		groups = DataManipulator.split_data_randomly(original_data, num_groups)
	else:
		#Use more complex split
		groups = DataManipulator.split_data_randomly_accounting_for_class(original_data, num_groups)

	return groups


#=============================
# MAIN PROGRAM
#=============================
def main():
	parser = argparse.ArgumentParser(description='Pre-process data by splitting it into 5 groups')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('separator', type=str, help='separator for data')
	parser.add_argument('num_groups', type=str, help='number of groups to split data')
	parser.add_argument('-o', action='store_true', help='output results to file')
	parser.add_argument('-r', action='store_true', help='totally random groups, may have disproportionate number of a given class')
	args = parser.parse_args()
	print(args)
	file_path = args.file_path
	separator = args.separator
	num_groups = args.num_groups
	is_random = args.r
	output_to_file = args.o

	groups = split_data(file_path, is_random, num_groups, separator)

	for counter, group in enumerate(groups):
		if output_to_file:
			group_file_name = "data_" + str(counter)
			print('writing file: ',   group_file_name)
			print(group_file_name)
			data_as_list = group.tolist()
			with open(group_file_name, 'w') as output_file:
				writer = csv.writer(output_file)
				for row in data_as_list:
					writer.writerow(row)
			#group = group.astype(str)
			#np.savetxt(group_file_name, group, delimiter=separator)
		print('group: ', counter)
		print('lenth: ', len(group))
		print(group)


if __name__ == '__main__':
	main()
