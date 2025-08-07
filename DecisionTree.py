import sys
import math
import numpy as np
import pandas as pd

"""
Decision Tree
In the command prompt syntax as
python3 DecisionTree.py datafile.csv
"""

class Node():
    """
    A class representing a node in a decision tree.
    """

    def __init__(self, mark="F",feature=None, left=None, right=None, gain=0, value=None, lorr= "", child =None):
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
        self.lorr = lorr
        self.child = child
        self.mark = mark
        

class DecisionTree():
    """
    A decision tree classifier for binary classification problems.
    initalise the Decision Tree.
    """
    min_samples = 0
    max_depth = 0

    def __init__(self, dataset, min_samples=2, max_depth=2):
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

        m =np.count_nonzero(y == 0)
        n=np.count_nonzero(y == 1)

        calculate_m = m/(m+n)
        calculate_n = n/(m+n)
        if calculate_m == 0 or calculate_n ==0:
            return 0
        ent = -(calculate_m*math.log(calculate_m))-(calculate_n*math.log(calculate_n))

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
        Hparent = self.entropy(parent)
        Hleft = self.entropy(left)
        Hright = self.entropy(right)
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
        feature = range(num_features)
        entropy = 0
        childleft = None
        childright = None
        num = 0
        for feature_index in feature:
            #TODO get the feature values
            value = dataset[:, feature_index]
            # get left and right datasets
            left_dataset, right_dataset = self.split_data(dataset, feature_index)
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
                    entropy = self.entropy(y)
                    num = feature_index
                    childleft = left_dataset
                    childright = right_dataset
                    
            feature_index+=1
        direct = {
            "featureIndex": num ,
            "entropy": entropy,
            "gain": best_split_gain,
            "left": childleft ,
            "right": childright,
        }

        return direct
    
    def find_instance(self, X, feature):
        value = feature
        count0= 0
        count1 = 0
        for row in X:
            if row[feature] == value:
                count0+=1
            else:
                count1+=1
        return count0, count1
    
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
        samplesnum, featuresnum = dataset.shape

        leaf_value = samplesnum
        
        if current_depth > self.max_depth:
            return Node(value=leaf_value, mark="F")
        if samplesnum < 2:
            return Node(value=leaf_value, mark= "F")
        
        bestvalues= self.best_split(dataset, samplesnum, featuresnum)

        if bestvalues['gain'] < 0.00001:
            return Node(value=leaf_value, mark = "F")
        
        current_depth +=1
        l1, l2 = self.find_instance(bestvalues['left'], bestvalues['featureIndex'])

        left_child = self.build_tree( bestvalues['left'] , current_depth )
        left_child.lorr = "{0: "+str(l1)+ " , 1: "+ str(l2)+" }"

        l1, l2 = self.find_instance(bestvalues['right'], bestvalues['featureIndex'])
        right_child = self.build_tree(bestvalues['right'] , current_depth )
        right_child.lorr = "{0: "+str(l1)+ " , 1: "+ str(l2)+" }"
        left_child.child = 0
        right_child.child = 1

        return Node(feature=bestvalues['featureIndex'], left=left_child, right=right_child, gain=bestvalues['gain'], value=bestvalues['entropy'], mark = "s")

    def fit(self, X):
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        """
        #dataset = np.concatenate((X, y), axis=1)  
        root = self.build_tree(X)
        return root

    def printTree(self, X):

        #print(X.feature," ", X.gain, " ", X.child," ", X.value, " ", X.left," ", X.right)
        prev = ""
        depth = ""
        feat = 0
        return self.recursive(X, depth, prev, feat)
    
    def recursive(self, X, depth, prev, feat):
        if X is None:
            return 
        depth = str(depth) + "  "
        feat = feat+ 1

        if X.mark == "F":
            prev = "\n" + depth + X.lorr
            return prev
        elif X.mark == "s":
            prev = "\n"+ depth + "feature "+ str(X.feature)+ " (IG: "+ str(X.gain)+ ", Entropy: "+ str(X.value)+")"
        
        prev= prev+"\n"+depth +"--feature "+ str(X.feature)+ " == 0 --"
        
        releft = self.recursive(X.left, depth, prev, feat)
        if releft is None:
            return 
        prev = prev + releft
        
        prev= prev +"\n"+depth +"--feature "+ str(X.feature)+ " == 1 --" 

        reright = self.recursive(X.right, depth, prev, feat)
        if reright is None:
            return 
        prev = prev + reright

        return prev
    
"""
running the code
"""
if len(sys.argv) == 3:
    print(sys.argv[0], " ", sys.argv[1])
    df = pd.read_csv(sys.argv[1])
    run = DecisionTree(df)
    nump = df.to_numpy()
    tree = run.fit(nump)
    pred = run.printTree(tree)
    print(pred)
    with open(sys.argv[2], "a") as f:
        f.write(pred)
        
else:
    print("There is no datafile or too many files")