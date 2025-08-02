import sys
import math
import numpy as np

"""
Decision Tree
In the command prompt syntax as
python3 DecisionTree.py datafile.csv
"""

class Node():
    """
    A class representing a node in a decision tree.
    """
    leaf = "leaf {" 

    def __init__(self, feature=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None.
            gain: The gain of the split. Defaults to None.
            value: If this node is a leaf node, this attribute represents the predicted value
                for the target variable. Defaults to None.
        """
        self.feature = feature
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
        

class DecisionTree():
    """
    A decision tree classifier for binary classification problems.
    initalise the Decision Tree.
    """


    def __init__(self, dataset , min_samples=2, max_depth=2):
        """
        Constructor for DecisionTree class.
        initialise
        all functions within the class is hidden

        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decision tree.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.dataset = dataset

        def split_data(self, dataset, feature):
            """
            Splits the given dataset into two datasets based on the given feature and threshold.
    
            Parameters:
                dataset (ndarray): Input dataset.
                feature (int): Index of the feature to be split on.
    
            Returns:
                left_dataset (ndarray): Subset of the dataset with values equal to the chosen category.
                right_dataset (ndarray): Subset of the dataset with values not equal to the chosen category.
            """
            # Create empty arrays to store the left and right datasets
            left_dataset = []
            right_dataset = []
        
            # Loop over each row in the dataset and split based on the given feature
            value = feature

            for row in dataset:
                if row[feature] == value:
                    left_dataset.append(row)
                else:
                    right_dataset.append(row)

            # Convert the left and right datasets to numpy arrays and return
            left_dataset = np.array(left_dataset)
            right_dataset = np.array(right_dataset)
            return left_dataset, right_dataset

        def entropy(self, y):
            """
            Computes the entropy of the given label values.
    
            Parameters:
                y (ndarray): Input label values.
    
            Returns:
                entropy (float): Entropy of the given label values.
            """
            
            m = 0
            n = 0
            # counts the amount of each attribute within the list y
            for num in y:
                if num == y[0]:
                    m +=1
                elif num != y[0] :
                    n +=1
        
            calculate_m = m/(m+n)
            calculate_n = n/(m+n)
            ent = (-calculate_m*math.log(calculate_m))-(calculate_n*math.log(calculate_n))

            # Return the final entropy value
            return ent
    
        def information_gain(self, parent, left, right):
            """
            Computes the information gain from splitting the parent dataset into two datasets.
    
            Parameters:
                parent (ndarray): Input parent dataset.
                left (ndarray): Subset of the parent dataset after split on a feature.
                right (ndarray): Subset of the parent dataset after split on a feature.
    
            Returns:
                information_gain (float): Information gain of the split.
            """

            #first find entropy of parent,left and right
            Hparent = self.entropy(self,parent)
            Hleft = self.entropy(self, left)
            Hright = self.entropy(self, right)
            wparent = len(parent)
            wleft = len(left)/wparent
            wright = len(right)/wparent

            information_gain = Hparent - ((wleft*Hleft)+(wright*Hright))

            return information_gain
        
        def best_split(self, dataset, num_samples, num_features):
            """
            Finds the best split for the given dataset.
    
            Args:
            dataset (ndarray): The dataset to split.
            num_samples (int): The number of samples in the dataset.
            num_features (int): The number of features in the dataset.
    
            Returns:
            dict: A dictionary with the best split feature index, threshold, gain, 
                  left and right datasets.
            """
            best_split_gain=-1
            value = 0
            feature = list(range(num_features))

            for feature_index in range(num_features):
                #TODO get the feature values
                value = feature[feature_index]
                # get left and right datasets
                left_dataset, right_dataset = self.split_data(dataset, feature[feature_index])
                # check if either datasets is empty
                if len(left_dataset) and len(right_dataset):
                    #get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    # compute information gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if conditions are met
                    if information_gain > best_split_gain:
                        #TODO: update the best_split_gain and the corresponding feature and threshold
                        best_split_gain = information_gain
                        feature_index+=1
                        
                        
                direct = {
                    "featureIndex": feature_index ,
                    "threshold":  0,
                    "gain": information_gain ,
                    "left": left_dataset ,
                    "right": right_dataset
                }

                return direct
            
        """
        After finding best split then build the tree
        """
        def build_tree(self, dataset, current_depth=0):
            """
            Recursively builds a decision tree from the given dataset.
    
            Args:
            dataset (ndarray): The dataset to build the tree from.
            current_depth (int): The current depth of the tree.
    
            Returns:
            Node: The root node of the built decision tree.
            """ 
            #TODO
            # return leaf node value
            return Node(value=leaf_value)

        def fit(self, X, y):
            """
            Builds and fits the decision tree to the given X and y values.
    
            Args:
            X (ndarray): The feature matrix.
            y (ndarray): The target values.
            """
            dataset = np.concatenate((X, y), axis=1)  
            self.root = self.build_tree(dataset)
        
        def predict(self, X):
            """
            Predicts the class labels for each instance in the feature matrix X.
    
            Args:
            X (ndarray): The feature matrix to make predictions for.
    
            Returns:
            list: A list of predicted class labels.
            """
            #TODO
            return predictions
        
"""
running the code
"""
if len(sys.argv) == 2:
    print(sys.argv[0], " ", sys.argv[1])
    run = DecisionTree(sys.argv[1],2,2)

else:
    print("There is no datafile or too many files")