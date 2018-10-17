#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	ClassificationTree class

import numpy as np
import pandas as pd
import argparse
import operator
import random
import math
import copy
from base_model3 import BaseModel

#=============================
# TreeNode
#
# - Class to encapsulate a tree nodek
#	- feature_id = column idx of data chosen
#		- Note at a leaf node, data[0] == class val 
#	- isLeaf = leaf node
#=============================
class TreeNode:
	def __init__(self, data, feature_id, isLeaf):
		self.data = data
		self.feature_id = feature_id
		#self.feature = feature #actual feature chosen for this node
		self.isLeaf = isLeaf
		#self.feature_id = feature_id #id of the feature in the original data - not sure it's useful
		#self.parent = parent #might be extra!
		self.children = dict()

	def get_classification(self):
		# get the majority class of the data @ this node
		#		- will provide easy way to prune!
		#			- can "fake" a leaf node
		classes = [self.data[-1] for row in self.data]
		unique_classes = set(classes)
		number_unique_classes = len(unique_classes)
		#Is this a fake or real leaf node
		if (number_unique_classes == 1):
			#True leaf node - only 1 class
			return self.data[0][-1]
		else:
			winning_class = unique_classes[0]
			winning_class_count = 0
			for unique_class in unique_classes:
				unique_class_count = classes.count(unique_class)
				if unique_class_count > winning_class_count:
					winning_class = unique_class
					winning_class_count = unique_class_count
			return winning_class


#=============================
# ClassificationTree
#
# - Class to encapsulate a tree classification decision model for 
#=============================
class ClassificationTree(BaseModel) :

	def __init__(self, data):
		BaseModel.__init__(self, data)

	#=============================
	# train()
	#
	#	- train on the data set
	#=============================
	def train(self):
		features = list(range(len(self.data[0])-1)) #don't include the class col
		#print(features)
		self.tree = self.build_tree(self.data, features, 0)
		return self.tree

	#=============================
	# build_tree()
	#	- builds the internal classification tree
	#		- called recursively
	#=============================
	def build_tree(self, data, features_available, recursive_depth):
		recursive_depth = recursive_depth + 1

		#Sanity check
		if (recursive_depth > 100):
			print('Runaway Recursion!, exceeding max recursive depth', + recursive_depth)
			return

		#Class stats - if they're all the same, this is leaf and recursion done
		#TODO: centralize this code? (tree node uses)
		classes = [row[-1] for row in data]
		print('classes')
		print(classes)
		unique_classes = set(classes)
		print('unique_classes')
		print(unique_classes)
		#If there is only one class, you're done!
		number_unique_classes = len(unique_classes)
		print('number_unique_classes')
		print(number_unique_classes)
		if (number_unique_classes == 1):
			isLeaf = True
			leaf_node = TreeNode(data, -1, isLeaf)
			return leaf_node

		#Winning feature based on highest gain ratio
		best_feature = self.get_best_feature(data, features_available)
		print('best_feature chosen this time: ', best_feature)

		#update features available by removing feature chosen
		features_available.remove(best_feature)
		print('features remaining:')
		print(features_available)

		#Create this node from the winning feature (column index)
		isLeaf = False
		tree_node = TreeNode(data, best_feature, isLeaf)

		#Get the feature values which will be our edges/children
		feature_values = [row[best_feature] for row in data]
		print('chosen feature_values')
		print(feature_values)
		unique_feature_values = set(feature_values)
		print('unique_feature_values')
		print(unique_feature_values)

		print('RETURNING before the recursion insanity')
		return
		#Create sub trees (recursively)
		for feature_value in unique_feature_values:
			#TODO: handle numeric vs. category features
			feature_value_data = [row for row in data if row[best_feature] == feature_value]
			print('feature_value_data')
			print(feature_value_data)
			tree_node.children[feature_value] = build_tree(
					feature_value_data, features_available, recursive_depth)

		return tree_node

	#=============================
	# get_best_feature()
	#=============================
	def get_best_feature(self, data, features_available):
		#Information gain / information value
		best_feature_performance = -1
		best_feature = -1
		for feature in features_available:
			print('feature (column) under examination')
			print(feature)
			feature_performance = self.calculate_gain_ratio(data, feature)
			print('gain ratio ', feature_performance, ' for feature ', feature)
			if feature_performance > best_feature_performance:
				best_feature = feature
				best_feature_performance = feature_performance
			print('best feature so far: ', best_feature, 'w/ ratio: ', best_feature_performance)
		return best_feature
	
	#=============================
	# calculate_gain_ratio()
	#=============================
	def calculate_gain_ratio(self, data, feature):
		#Information gain / information value
		info_gain = self.calculate_information_gain(data, feature)
		print('information gain ', info_gain, 'for feature ', feature)
		info_value = self.calculate_information_val(data, feature)
		print('info value ', info_value, 'for feature ', feature)
		return float(info_gain / info_value)

	#=============================
	# calculate_information_gain()
	#=============================
	def calculate_information_gain(self, data, feature):
		information = self.calculate_information(data)
		print('information (total) ', information, 'for feature', feature)
		entropy = self.calculate_entropy(data, feature)
		print('entropy (total) ', entropy, 'for feature', feature)
		return float(information - entropy)

	#=============================
	# calculate_information()
	#=============================
	def calculate_information(self, data):
		total_info_val = 0
		classes = [row[-1] for row in data]
		num_class_vals = len(classes)
		unique_classes = set(classes)
		for a_class in unique_classes:
			number_of_a_class = classes.count(a_class)
			ratio = float(number_of_a_class / num_class_vals)
			class_info_val = ratio * math.log(ratio, 2)
			total_info_val = total_info_val + class_info_val

		return float(-1.0 * total_info_val)


	#=============================
	# calculate_entropy()
	#=============================
	def calculate_entropy(self, data, feature):
		#TODO: centralize & do this once: repeated in calc entropy
		#TODO: handle numeric inputs (i.e. try splits at every possible split)
		feature_values = [row[feature] for row in data]
		print('feature_values')
		print(feature_values)
		unique_feature_values = set(feature_values)
		print('unique_feature_values')
		print(unique_feature_values)
		total_feature_entropy = 0
		total_num_class_values = len(data)
		for feature_value in unique_feature_values:
			#Get dataset for this feature:
			#TODO: if this partition is ever EMPTY totally disqualify this feature
			feature_data = [row for row in data if row[feature] == feature_value]
			feature_data_info = self.calculate_information(feature_data)
			class_values_in_feature_subset = len(feature_data)
			ratio = class_values_in_feature_subset/total_num_class_values
			total_feature_entropy = total_feature_entropy + float(ratio * feature_data_info)
		#TODO: account for possibly no feature being chosen?!

		return total_feature_entropy

	#=============================
	# calculate_information_value()
	#=============================
	def calculate_information_val(self, data, feature):
		#TODO: centralize & do this once: repeated in calc entropy
		#TODO: handle numeric inputs (i.e. try splits at every possible split)
		feature_values = [row[feature] for row in data]
		unique_feature_values = set(feature_values)
		print('unique_feature_values')
		print(unique_feature_values)
		total_info_value = 0.0
		total_class_values = len(data)
		for feature_value in unique_feature_values:
			#Get dataset for this feature:
			feature_data = [row for row in data if row[feature] == feature_value]
			class_values_in_feature_subset = len(feature_data)
			ratio = float(class_values_in_feature_subset/total_class_values)
			total_info_value = float(total_info_value + (ratio * math.log(ratio, 2)))

		return float(-1 * total_info_value)

	#=============================
	# validate()
	#
	#	- validate the data, i.e. prune or optimize for generalization
	#=============================
	def validate(self, validation_data):
		#TODO: pruning here
		#	- 1. calculate overall performance
			#- 2. recursively traverse where for each node
				#- set each child to "isLeaf" thereby triggering majority calculation 
				#- save performance value
				#- immediately return once you've got a better performance value than the original tree
			#- 3. Calculate performance for NEW tree, repeat above
		return

	#=============================
	# test()
	#
	#	- test the model 
	#
	#@param		test_data to evaluat	
	#@return	value of performance as percent class error
	#=============================
	def test(self, test_data):
		#TODO: Traverse the tree
		#	- will require separate logic for category vs. numeric
		#	- will be recursive?
		total_classifications = 0
		correct_classifications = 0
		#Analyze each row separately
		for row in test_data:
			node = self.tree
			while node.isLeaf != True:
				value = row[node.feature_id]
				'''
				#TODO: implement numeric traverse
				if value.isNumeric():
					keys = node.children.keys
					next_child_key = get_closest_val(keys, value)
					next_node = node.children[next_child_key]
				else:
				'''
				node = node.children[value]
				if node == NULL: 
					print('Never seen this val before')
					break
			
			model_classification = node.get_classification()
			data_classification = row[-1]
			if (model_classification == data_classification):
				correct_classifications = correct_classifications + 1
			total_classifications = total_classifications + 1

		return float( (correct_classifications / total_classifications) * 100)


#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing test model')

	print()
	print('TEST 1: dummy data')
	print('NOTE TO SELF: the example from class uses INFORMATION GAIN, NOT GAIN RATIO - possibly different results')
	print('input data1:')
	#TODO: turn this into dataframe
	#test_data = pd.DataFrame([[0, 1, -1], [0, 1, -1],[0, 1, -1]])
	test_data = [ \
			['Sunny', 'Hot', 'High', 'False', 'N'],
			['Sunny', 'Hot', 'High', 'True', 'N'],
			['Overcast', 'Hot', 'High', 'False', 'P'],
			['Rainy', 'Mild', 'High', 'False', 'P'],
			['Rainy', 'Cool', 'Normal', 'False', 'P'],
			['Rainy', 'Cool', 'Normal', 'True', 'N'],
			['Overcast', 'Cool', 'Normal', 'True', 'P'],
			['Sunny', 'Mild', 'High', 'False', 'N'],
			['Sunny', 'Cool', 'Normal', 'False', 'P'],
			['Rainy', 'Mild', 'Normal', 'False', 'P'],
			['Sunny', 'Mild', 'Normal', 'True', 'P'],
			['Overcast', 'Mild', 'High', 'True', 'P'],
			['Overcast', 'Hot', 'Normal', 'False', 'P'], \
			['Rainy', 'Mild', 'High', 'True', 'N'] \
			]
	#test_data2 = #TODO: something with numerics?!

	for line in test_data:
		print(line)
	print()

	classification_tree = ClassificationTree(test_data)
	classification_tree.train()
	#validation_data = test_data
	#validatedTree = test_model.validate() #Should be 0 pruning....
	percent_accurate = test_model.test(test_data)
	print('percent_accurate')
	print(percent_accurate)


if __name__ == '__main__':
	main()
