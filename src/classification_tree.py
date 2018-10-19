#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			10/16/2018
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
#	- #NOTE, children dict keys are "less_than" and "greater_than" for continuous values
#=============================
class TreeNode:
	def __init__(self, data, feature_id, isLeaf, split_value=None):
		self.data = data #All of the data @ this node - useful for pruning!
		self.feature_id = feature_id #feature chosen
		self.split_value = split_value #Continuous values need to know where the binary split occured
		self.isLeaf = isLeaf #Leaf nodes are effectively classification nodes
		#Children populated via build_tree
		self.children = dict() #NOTE, for continous values "less_than" & "greater_than" will be dict keys

	def get_classification(self):
		# get the majority class of the data @ this node
		#		- will provide easy way to prune!
		#			- can "fake" a leaf node
		classes = [row[-1] for row in self.data]
		#print('classes in node data')
		#print(classes)
		unique_classes = set(classes)
		#print('unique_classes in node data')
		#print(unique_classes)
		number_unique_classes = len(unique_classes)
		#Is this a fake or real leaf node
		if (number_unique_classes == 1):
			#True leaf node - only 1 class
			return classes[0]
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

	LESS_THAN = 'less_than'
	GREATER_THAN = 'greater_than'
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
		#print('classes')
		#print(classes)
		unique_classes = set(classes)
		#print('unique_classes')
		#print(unique_classes)
		#If there is only one class, you're done!
		number_unique_classes = len(unique_classes)
		#print('number_unique_classes')
		#print(number_unique_classes)
		if (number_unique_classes == 1):
			isLeaf = True
			leaf_node = TreeNode(data, -1, isLeaf)
			return leaf_node

		#Winning feature based on highest gain ratio
		best_feature= self.get_best_feature(data, features_available)
		example_best_feature_value = data[0][best_feature[0]]

		#CONTINUOUS FEATURE
		if (example_best_feature_value.isdigit()): 
			#TODO: continous values
			split_value = best_feature[1] 
			best_feature = best_feature[0] 
			print('best_feature ', best_feature)
			print('split_value ', split_value)

			#Create this node from the winning feature (column index)
			isLeaf = False
			tree_node = TreeNode(data, best_feature, isLeaf, split_value)

			#Less_than continuous values
			feature_value_less_than_data = \
				[row for row in data if float(row[best_feature]) <= split_value]
			features_available_less_than = copy.deepcopy(features_available)
			feature_values_less_than = [row[best_feature] for row in feature_value_less_than_data]
			unique_feature_values_less_than = set(feature_values_less_than)

			#If we are down to a partition of size 1 after a split, remove that feature from consideration
			if (len(unique_feature_values_less_than) <= 1):
				features_available_less_than.remove(best_feature)

			tree_node.children[ClassificationTree.LESS_THAN] = self.build_tree(
				feature_value_less_than_data, features_available_less_than, recursive_depth)

			#greater_than continuous values
			feature_value_greater_than_data = \
				[row for row in data if float(row[best_feature]) > split_value]
			features_available_greater_than = copy.deepcopy(features_available)
			feature_values_greater_than = [row[best_feature] for row in feature_value_greater_than_data]
			unique_feature_values_greater_than = set(feature_values_greater_than)

			#If we are down to a partition of size 1 after a split, remove that feature & create leaf node as child
			if (len(unique_feature_values_greater_than) <= 1):
				features_available_greater_than.remove(best_feature)

			tree_node.children[ClassificationTree.GREATER_THAN] = self.build_tree(
				feature_value_greater_than_data, features_available_greater_than, recursive_depth)

		#CATEGORICAL FEATURE
		else: 
			best_feature = best_feature[0] #second value is pointless
			#print('best_feature chosen this time: ', best_feature)

			#update features available by removing feature chosen
			features_available.remove(best_feature)
			#print('features remaining:')
			#print(features_available)

			#Create this node from the winning feature (column index)
			isLeaf = False
			tree_node = TreeNode(data, best_feature, isLeaf)

			#Get the feature values which will be our edges/children
			feature_values = [row[best_feature] for row in data]
			#print('chosen feature_values')
			#print(feature_values)
			unique_feature_values = set(feature_values)
			#print('unique_feature_values')
			#print(unique_feature_values)

			#Create sub trees (recursively)
			for feature_value in unique_feature_values:
				feature_value_data = [row for row in data if row[best_feature] == feature_value]
				#print('feature_value_data')
				#print(feature_value_data)
				tree_node.children[feature_value] = self.build_tree(
					feature_value_data, features_available, recursive_depth)

		return tree_node

	#=============================
	# get_best_feature()
	#	- returns tuple:
	#		- CATEGORICAL: (best_feature, None)
	#		- CONTINUOUS: (best_feature, split_value)
	#=============================
	def get_best_feature(self, data, features_available):
		#Information gain / information value
		best_feature_performance = -1
		best_feature = -1
		best_feature_split = None #None for CATEGORICAL data but VALID for CONTINUOUS data
		for feature in features_available:
			#print('feature (column) under examination')
			#print(feature)
			feature_performance = self.calculate_gain_ratio(data, feature)
			#print('gain ratio ', feature_performance, ' for feature ', feature)
			if feature_performance[0] > best_feature_performance:
				best_feature = feature
				best_feature_performance = feature_performance[0]
				best_feature_split = feature_performance[1]
			#print('best feature so far: ', best_feature, 'w/ ratio: ', best_feature_performance)
		return (best_feature, best_feature_split)
	
	#=============================
	# calculate_gain_ratio()
	#=============================
	def calculate_gain_ratio(self, data, feature):
		split_value = None
		best_split_value = 0
		best_feature_gain_ratio = 0

		#CONTINUOUS values try all splits and take the best one
		if (data[0][feature].isdigit() == True):
			#Get values for feature
			feature_values = [row[feature] for row in data]
			#Get unique values for testing splits
			unique_feature_values_set = set(feature_values)
			unique_feature_values = list(unique_feature_values_set)
			#Sort values for feature
			unique_feature_values.sort()
			#keep track of best split_value
			for feature_idx in ( range(len(unique_feature_values) - 1) ): #Don't need the last value
				#get midpoint value & next value
				print('unique_feature_values')
				print(unique_feature_values)
				split_value = float((float(unique_feature_values[feature_idx]) + float(unique_feature_values[feature_idx+1])) / 2.0)
				#calculate info_gain_ratio
				#Information gain / information value
				info_gain = self.calculate_information_gain(data, feature, split_value)
				#print('information gain ', info_gain, 'for feature ', feature)
				info_value = self.calculate_information_val(data, feature, split_value)
				#print('info value ', info_value, 'for feature ', feature)
				gain_ratio = float(info_gain / info_value)
				#Keep track of largest info_gain_ratio & "split value"
				if gain_ratio > best_feature_gain_ratio:
					best_feature_gain_ratio = gain_ratio
					best_feature_split = split_value

		#CATEGORICAL Feature
		else: 
			#Information gain / information value
			info_gain = self.calculate_information_gain(data, feature)
			#print('information gain ', info_gain, 'for feature ', feature)
			info_value = self.calculate_information_val(data, feature)
			#print('info value ', info_value, 'for feature ', feature)
			best_feature_gain_ratio = float(info_gain / info_value)
		return (best_feature_gain_ratio, split_value)

	#=============================
	# calculate_information_gain()
	#=============================
	def calculate_information_gain(self, data, feature, split_value=None):
		information = self.calculate_information(data)
		#print('information (total) ', information, 'for feature', feature)
		entropy = self.calculate_entropy(data, feature, split_value)
		#print('entropy (total) ', entropy, 'for feature', feature)
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
	def calculate_entropy(self, data, feature, split_value=None):
		#TODO: centralize & do this once: repeated in calc entropy
		total_feature_entropy = 0
		total_num_class_values = len(data)

		#CATEGORICAL Values
		if split_value == None:
			feature_values = [row[feature] for row in data]
			#print('feature_values')
			#print(feature_values)
			unique_feature_values = set(feature_values)
			#print('unique_feature_values')
			#print(unique_feature_values)
			for feature_value in unique_feature_values:
				#Get dataset for this feature:
				feature_data = [row for row in data if row[feature] == feature_value]
				#if this partition is ever EMPTY totally disqualify this feature
				if len(feature_data) == 0:
					total_feature_entropy = 999; #Should ensure this feature is NOT chosen
					return total_feature_entropy
				feature_data_info = self.calculate_information(feature_data)
				class_values_in_feature_subset = len(feature_data)
				ratio = class_values_in_feature_subset/total_num_class_values
				total_feature_entropy = total_feature_entropy + float(ratio * feature_data_info)
			#TODO: account for possibly no feature being chosen?!

		#CONTINUOUS Values
		else: #CONTINUOUS Values
			#Get partitions of data, less-than & greater-than
			less_than_feature_data = [row for row in data if float(row[feature]) <= split_value]
			greater_than_feature_data = [row for row in data if float(row[feature]) > split_value]

			#if either partition is ever EMPTY totally disqualify this split
			if len(less_than_feature_data) == 0 or len(greater_than_feature_data) == 0:
				total_feature_entropy = 999; #Should ensure this feature is NOT chosen
				return total_feature_entropy

			#Less Than Partition Entropy
			less_than_feature_data_info = self.calculate_information(less_than_feature_data)
			less_than_class_values_in_feature_subset = len(less_than_feature_data)
			less_than_ratio = less_than_class_values_in_feature_subset/total_num_class_values
			total_feature_entropy = total_feature_entropy + float(less_than_ratio * less_than_feature_data_info)

			#Greater Than Partition Entropy
			greater_than_feature_data_info = self.calculate_information(greater_than_feature_data)
			greater_than_class_values_in_feature_subset = len(greater_than_feature_data)
			greater_than_ratio = greater_than_class_values_in_feature_subset/total_num_class_values
			total_feature_entropy = total_feature_entropy + float(greater_than_ratio * greater_than_feature_data_info)

		return total_feature_entropy

	#=============================
	# calculate_information_value()
	#=============================
	def calculate_information_val(self, data, feature, split_value=None):
		#TODO: centralize & do this once: repeated in calc entropy
		total_info_value = 0.0
		total_class_values = len(data)

		#Continuous Values
		if split_value != None:
			#get Less than & greater than feature data
			less_than_feature_data = [row for row in data if float(row[feature]) <= split_value]
			greater_than_feature_data = [row for row in data if float(row[feature]) > split_value]

			#if either partition is ever EMPTY totally disqualify this split
			if len(less_than_feature_data) == 0 or len(greater_than_feature_data) == 0:
				total_feature_entropy = 999; #Should ensure this feature is NOT chosen
				return total_feature_entropy

			#Less Than Partition Entropy
			less_than_class_values_in_feature_subset = len(less_than_feature_data)
			less_than_ratio = float(less_than_class_values_in_feature_subset/total_class_values)
			total_info_value = float(total_info_value + (less_than_ratio * math.log(less_than_ratio, 2)))

			#Greater Than Partition Entropy
			greater_than_class_values_in_feature_subset = len(greater_than_feature_data)
			greater_than_ratio = float(greater_than_class_values_in_feature_subset/total_class_values)
			total_info_value = float(total_info_value + (less_than_ratio * math.log(greater_than_ratio, 2)))

		#CATEGORICAL Values
		else:
			feature_values = [row[feature] for row in data]
			unique_feature_values = set(feature_values)
			#print('unique_feature_values')
			#print(unique_feature_values)
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
			print('testing row: ', row)
			node = self.tree
			#print('root node:')
			#print(test_data[0][node.feature_id])
			while node.isLeaf != True:
				value = row[node.feature_id]

				#CONTINUOUS Value
				if value.isdigit() == True:
					if float(value) <= node.split_value:
						node = node.children[ClassificationTree.LESS_THAN]
					else:
						node = node.children[ClassificationTree.GREATER_THAN]

				#CATEGORICAL Value
				else:
					prev_node = node
					node = node.children[value]

				if node is None:
					print('Never seen this value ', value, 'before, cant classify traverse tree')
					node = prev_node
					break
			
			model_classification = node.get_classification()
			print('model classification: ', model_classification)
			data_classification = row[-1]
			print('data classification: ', data_classification)
			if (model_classification == data_classification):
				correct_classifications = correct_classifications + 1
			total_classifications = total_classifications + 1
			#print('total_classifications')
			#print(total_classifications)

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
	test_data2 = [ \
			['Sunny', '3', 'High', '1', 'N'],
			['Sunny', '3', 'High', '2', 'N'],
			['Overcast', '3', 'High', '1', 'P'],
			['Rainy', '2', 'High', '1', 'P'],
			['Rainy', '1', 'Normal', '1', 'P'],
			['Rainy', '1', 'Normal', '2', 'N'],
			['Overcast', '1', 'Normal', '2', 'P'],
			['Sunny', '2', 'High', '1', 'N'],
			['Sunny', '1', 'Normal', '1', 'P'],
			['Rainy', '2', 'Normal', '1', 'P'],
			['Sunny', '2', 'Normal', '2', 'P'],
			['Overcast', '2', 'High', '2', 'P'],
			['Overcast', '3', 'Normal', '1', 'P'], \
			['Rainy', '2', 'High', '2', 'N'] \
			]

	#for line in test_data:
		#print(line)
	#print()
	for line in test_data2:
		print(line)
	print()

	#classification_tree = ClassificationTree(test_data)
	classification_tree = ClassificationTree(test_data2)
	classification_tree.train()
	#validation_data = test_data
	#validatedTree = test_model.validate() #Should be 0 pruning....
	#percent_accurate = classification_tree.test(test_data)
	percent_accurate = classification_tree.test(test_data2)
	print()
	print('Model Accuracy:', percent_accurate, '%')


if __name__ == '__main__':
	main()
