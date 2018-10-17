#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	BaseModel class

import numpy as np
import pandas as pd
import argparse
import operator
import copy

#=============================
# BaseModel
#
# - Base class of a model / classifier 
#=============================
class BaseModel:

	def __init__(self, data):
		self.data = data

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		print('Base Class training')

	#=============================
	# validate()
	#
	#	- validate the data, i.e. prune or optimize for generalization
	#=============================
	def train(self):
		print('Base Class validation')


	#=============================
	# test()
	#
	#	- test the model 
	#
	#@param		test_data	optional, can provide other data set to use
	#@return				value of performance
	#=============================
	def test(self, test_data=-1):
		print('Base Class training')
		if (test_data == -1):
			print('No input test_data provided, testing performance on my training data')

		print('test the base model, no such thing!')
		return -1;
	
#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing base model - no such thing')


''' COMMENTED OUT FOR SUBMITTAL
if __name__ == '__main__':
	main()
	'''
