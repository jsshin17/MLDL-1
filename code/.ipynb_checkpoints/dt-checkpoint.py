"""
A starting code for a basic decision tree classifier for
binary classification.  This implementation should be based on the
minimum classification error heuristic.
"""

from numpy import *
from scipy import stats
from binary import *
import util

class DT(BinaryClassifier):
    """
    This class is for the decision tree implementation.  
    It has a partial implementation for the tree data structure. 
    This class also has a function to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)
        
        self.isLeaf = True
        self.label  = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth*2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth*2)) + "Branch " + repr(self.feature) + "\n" + \
                      self.left.displayTree(depth+1) + \
                      self.right.displayTree(depth+1)

    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        if self.isLeaf:
            return str(self.label)
        else:
            if(X[self.feature])<0.5:
                return self.left.predict(X)
            else:
                return self.right.predict(X)



    def trainDT(self, X, Y, maxDepth, used):
        """
        recursively build the decision tree
        """

        # get the size of the data set
        N,D = X.shape

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True

            self.label = util.mode(Y)   # do not have to make any decision

        else:
            # we need to find a feature to split on
            bestFeature = -1     # which feature has lowest error
            bestError   = N      # the number of errors for this feature
            for d in range(D):
                # have we used this feature yet
                if d in used:
                    continue

                # suppose we split on this feature; what labels
                # would go left and right?
                leftY  =  Y[X[:, d]<0.5]   # left for feature that are less than 0.5

                rightY =  Y[X[:, d]>=0.5]


                # we'll classify the left points as their most
                # common class and ditto right points.  our error
                # is the how many are not their mode.
                error = size((leftY!=util.mode(leftY)).nonzero()) + size((rightY!=util.mode(rightY)).nonzero())


                # check to see if this is a better error rate
                if error <= bestError:
                    bestFeature = d
                    bestError   = error

            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label  = util.mode(Y)

            else:
                self.isLeaf  = False

                self.feature = bestFeature


                self.left  = DT({'maxDepth': maxDepth-1})
                self.right = DT({'maxDepth': maxDepth-1})
                # recurse on our children by calling
                #   self.left.trainDT(...) 
                # and
                #   self.right.trainDT(...) 
                # with appropriate arguments
                self.left.trainDT(X[X[:, self.feature] < 0.5], Y[X[:, self.feature] < 0.5], self.left.opts['maxDepth'], used + [self.feature, ])
                self.right.trainDT(X[X[:, self.feature] >= 0.5], Y[X[:, self.feature] >= 0.5], self.right.opts['maxDepth'], used + [self.feature, ])
                # For Chi-square pruning
                self.split = array([[size((Y[X[:, self.feature] < 0.5] == 1).nonzero()), size((Y[X[:, self.feature] < 0.5] == -1).nonzero())],
                                    [size((Y[X[:, self.feature] >= 0.5] == 1).nonzero()), size((Y[X[:, self.feature] >= 0.5] == -1).nonzero())]])

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X(:,5)>=0.5.  If you want the corresponding classes,
            say Y(X(:,5)>=0.5) and if you want the corresponding rows of X,
            say X(X(:,5)>=0.5,:)
            
          - I suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """

        self.trainDT(X, Y, self.opts['maxDepth'], [])



    def pruneByChiSquare(self, maxPvalue):
        """
        Prune your tree by chi-square testing with the hyperparameter of maxPvalue.
        """
        if self.isLeaf == False:
            self.right.pruneByChiSquare(maxPvalue)
            self.left.pruneByChiSquare(maxPvalue)
            if self.right.isLeaf and self.left.isLeaf:
                if stats.chi2_contingency(self.split)[1] > maxPvalue:
                    self.isLeaf = True
                    self.label = 2*(sum(self.split[:, 0]) > sum(self.split[:, 1])) - 1


    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """
        
        return self

