#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Handle file input

import csv
import numpy as np
import pandas as pd
import argparse

#=============================
# FILE_MANAGER
#
#	- Contains methods for processing File I/O
#=============================
class FileManager:

	#=============================
	# GET_CSV_FILE_DATA_ARRAY()
	#
	#	- get csv file as python array
	#	- data type of objects will be float
	#
	#@return	array (lists)
	#=============================
	@staticmethod
	def get_csv_file_data_array(file_name, type='str'):
		#print('LOG: get_csv_file_data_array() START')
		all_data = list()
		with open(file_name, newline='') as csvfile:
			csv_reader = csv.reader(csvfile, delimiter=',')
			all_data = [data for data in csv_reader]

		if type == 'float':
			#convert this data to float from string
			data_as_float = []
			for vector in all_data:
				tmp_float_vector = list(map(float, vector))
				data_as_float.append(tmp_float_vector)
			return data_as_float

		#print(all_data)
		#print('LOG: get_csv_file_data_array() END')
		return all_data

	#=============================
	# GET_CSV_FILE_DATA_NUMPY()
	#
	#	- get csv file as numpy array
	#
	#@return	ndarray
	#=============================
	@staticmethod
	def get_csv_file_data_numpy(file_name, separator=','):
		#print('LOG: get_csv_file_data_numpy() START')
		all_data = np.genfromtxt(file_name, dtype=str, delimiter=separator)
		#print(all_data)
		#print('LOG: get_csv_file_data_numpy() END')
		return all_data

	#=============================
	# GET_CSV_FILE_DATA_PANDAS()
	#
	#	- get csv file as pandas structure
	#
	#@return	pandas structure
	#=============================
	@staticmethod
	def get_csv_file_data_pandas(file_name, separator=','):
		#print('LOG: get_csv_file_data_pandas() START')

		all_data = pd.read_csv(file_name, sep=separator, header=None)
		#print(all_data.values)
		#print('LOG: get_csv_file_data_pandas() END')
		return all_data

	#=============================
	# WRITE_2D_ARRAY_TO_CSV()
	#
	#	- get csv file as pandas structure
	#
	#@param		data		data, expected to be 2D matrix
	#@param		filename	name of file to create
	#=============================
	@staticmethod
	def write_2d_array_to_csv(data, file_name):
		with open(file_name, "w+") as csv_file:
			csv_writer = csv.writer(csv_file, delimiter=',')
			csv_writer.writerows(data)
		return

#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to test class "FileManager"')
	parser = argparse.ArgumentParser(description='Handle file input')
	parser.add_argument('file_path', type=str, help='full path to input file')
	args = parser.parse_args()
	file_path = args.file_path

	print()
	print('TEST 1: get normal csv file data')
	print('data:')
	print(FileManager.get_csv_file_data_array(file_path))

	print()
	print('TEST 2: get numpy csv file data')
	print('data:')
	print(FileManager.get_csv_file_data_numpy(file_path))

	print()
	print('TEST 3: get pandas csv file data')
	print('data:')
	print(FileManager.get_csv_file_data_pandas(file_path))


''' COMMENTED OUT FOR SUBMITTAL
if __name__ == '__main__':
	main()
	'''
