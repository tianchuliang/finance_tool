import numpy as np
import copy
from scipy.stats import mode
class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def build_tree(self, data):
        if self.verbose:
            print 'Building tree, data shapes: ', data.shape

        if data.shape[0] <= self.leaf_size:
            if self.verbose:
                print '     reached leaf_size', mode(data[:,-1])[0]
            return np.array([[np.nan, mode(data[:,-1])[0], np.nan, np.nan]], dtype=float)
        
        if len(np.unique(data[:,-1])) == 1:
            if self.verbose:
                print '     all label are the same', mode(data[:,-1])[0]
            return np.array([[np.nan, mode(data[:,-1])[0], np.nan, np.nan]], dtype=float)

        split_index = np.random.randint(data.shape[1] - 1)
        random_row_1 = np.random.randint(data.shape[0])
        random_row_2 = np.random.randint(data.shape[0])
        
        if self.verbose:
            print 'random rows: ', random_row_1, random_row_2
        
        split_val = np.mean(data[random_row_1, split_index] + data[random_row_2, split_index]) / float(2)
        
        if self.verbose:
            print 'split_val is: ', split_val

        while (data[data[:,split_index]<=split_val].shape[0] == 0 \
            or data[data[:,split_index]>split_val].shape[0] == 0):
            if self.verbose:
                print 'avoiding empty branch'
            split_index = np.random.randint(data.shape[1] - 1)                
            random_row_1 = np.random.randint(data.shape[0])
            random_row_2 = np.random.randint(data.shape[0])
            split_val = np.mean(data[random_row_1, split_index] + data[random_row_2, split_index]) / float(2)
        
        left_tree = self.build_tree(data[data[:,split_index]<=split_val])
        right_tree = self.build_tree(data[data[:,split_index]>split_val])
        
        if self.verbose:
            print 'left tree: ', left_tree, left_tree.shape[0]
            print 'right tree: ', right_tree, right_tree.shape[0]
        
        root = np.array([split_index, split_val, 1, left_tree.shape[0] + 1])
        sub_tree = np.vstack((root, left_tree))
        sub_tree = np.vstack((sub_tree, right_tree))

        return sub_tree

    def addEvidence(self, dataX, dataY):

        if self.verbose:
            print 'Adding evidence and invoking build_tree()'
        dataY_copy = copy.deepcopy(dataY)
        dataY_copy.shape = (dataY_copy.shape[0],1)
        dataX_copy = copy.deepcopy(dataX)

        self.tree = self.build_tree(np.hstack((dataX_copy, dataY_copy)))
        if self.verbose:
            print '*************Tree Constructed: ',self.tree

    def query(self,points):
        if self.verbose:
            print 'QUERYING:::::::::::::::::::::::::::::::::::::::::::::::'
        
        if self.verbose:
            print points.shape
        split_factors = self.tree[:,0]
        split_values = self.tree[:,1]
        left_branches = self.tree[:,2]
        right_branches = self.tree[:,3]

        tree_positions = np.arange(self.tree.shape[0])
        points_positions = np.arange(points.shape[0])
        predictions = np.zeros((points.shape[0]))

        finished = np.zeros((points.shape[0]))

        # progress contains cumulative tree-node-index
        progress = np.zeros((points.shape[0]), dtype = int)

        # while there is still unfinished (0) points
        while 0 in finished:

            if self.verbose:
                # raw_input("------------------------------------Press Enter to continue...")
                print 'exploring tree, current progress: ', progress
            # use progress, find corresponding split_indices
            points_split_indices = split_factors[progress] #dim: num_of_points * 1
            points_split_values = split_values[progress] #dim: num_of_points * 1
            if self.verbose:
                print 'points_split_indices: ', points_split_indices
                print 'points_split_values: ', points_split_values
            # use np.isnan() to find leaf-reaching nodes and update finished and predictions
            points_reached_leaves = np.isnan(points_split_indices) #dim: num_of_points * 1, type: Boolean
            if self.verbose:
                print 'points_reached_leaves: ', points_reached_leaves
            more_finished = 1*points_reached_leaves
            if self.verbose:
                print 'more_finished: ', more_finished
            finished = more_finished + finished
            if self.verbose:
                print 'finished: ', finished
            more_predicted = points_split_values * points_reached_leaves
            if self.verbose:
                print '==============>more_predicted: ', more_predicted
            predictions = more_predicted
            if self.verbose: 
                print 'predictions: ', predictions
            
            if 0 not in finished:
                break

            # use np.isnan() to also find non-terminating points
            points_unfinished = np.invert(points_reached_leaves)
            if self.verbose:
                print 'points_unfinished: ', points_unfinished

            if not points_unfinished[0]:
                points_unfinished_indices_first = np.array([])
            else:
                points_unfinished_indices_first = np.array([0])

            if self.verbose: 
                print 'points_unfinished_indices_first: ', points_unfinished_indices_first

            points_unfinished_indices_rest = (points_positions * points_unfinished)[1:]
            points_unfinished_indices_rest = set(points_unfinished_indices_rest)
            points_unfinished_indices_rest.discard(0)
            if self.verbose:
                print 'points_unfinished_indices_rest: ', points_unfinished_indices_rest
            points_unfinished_indices_rest = points_unfinished_indices_rest | set(points_unfinished_indices_first)
            points_unfinished_indices = np.array(list(points_unfinished_indices_rest))
            if self.verbose:
                print 'points_unfinished_indices: ', points_unfinished_indices
            
            # for non-terminating points, do split value comparisons and update progress
            points_unfinished_split_values = (points_split_values * points_unfinished)[points_unfinished_indices]
            if self.verbose:
                print 'points_unfinished_split_values: ', points_unfinished_split_values
            relevant_unfinished_column_indices = points_split_indices[points_unfinished_indices]
            relevant_unfinished_column_indices = relevant_unfinished_column_indices.astype(int)
            if self.verbose:
                print 'relevant_unfinished_column_indices: ', relevant_unfinished_column_indices
            points_unfinished_values = points[points_unfinished_indices, relevant_unfinished_column_indices]
            if self.verbose:
                print 'points_unfinished_values: ', points_unfinished_values

            right_values = right_branches[progress][points_unfinished_indices]
            if self.verbose:
                print 'right_values', right_values
            left_values = left_branches[progress][points_unfinished_indices]
            if self.verbose:
                print 'left_values', left_values

            go_left = (points_unfinished_values <= points_unfinished_split_values) * left_values
            
            go_right = (points_unfinished_values > points_unfinished_split_values) * right_values
            if self.verbose:
                print 'go_left: ', go_left
                print 'go_right: ', go_right

            addition = go_left + go_right

            if self.verbose: 
                print 'addition: ', addition
            progress[points_unfinished_indices] = progress[points_unfinished_indices] + addition

            if self.verbose:
                print 'progress: ', progress

        return predictions
