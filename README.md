# Coding Assignment 1: A naive classification and decision trees

## Introduction

The first coding assignment asks you to implement two classifiers to predict whether students are likely to take the AI (Artificial Intelligence) course (AI) or the CG (Computer Graphics) course (CG). 1) We begin by implementing some very simple predictors and We will implement a decision tree.

**Note**: we will use `Python 2.x` for the project. If you want to use `Python 3.x`, you have to modify some of the utility functions (but not much. It should be doable.).

We provide the code consisting of several Python files, some of which you will need to read and understand in order to complete the assignment, and some of which you can ignore. You will figure them out which one is which.

<!-- **Please start your coding assignment by accepting <a href="">a git classroom assignment</a>. It will provide all the codes and supporting files placed in your private repo.** -->

---

**Files you'll edit:**

`dumbClassifiers.py` contains "warm up" classifiers to get you used to our this code framework.

`dt.py` will be your decision tree implementation.

---

**Files you may want to review:**

`binary.py` A generic interface for binary classifiers or regressions.

`datasets.py` Datasets.

`util.py` Utility functions!

`runClassifier.py` A few wrapper functions for running classifiers: training, generating learning curves, etc.

`mlGraphics.py` Plotting commands.

---

**What to submit (Push to your github classroom):** 
- All of the python files listed above (under "Files you'll edit"). 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

<!-- **Evaluation:** Your code will be autograded for technical correctness. Please do not change the names of any provided functions or classes within the code, or you will wreak havoc on the autograder. However, the correctness of your implementation -- not the autograder's output -- will be the final judge of your score. If necessary, we will review and grade assignments individually to ensure that you receive due credit for your work. -->

**Academic Dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting Help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

<!-- **One more piece of advice:** if you don't know what a variable is, print it out. -->

---

### Warming Up to Classifiers (25%)

Let's begin our foray into classification by looking at some very simple classifiers. There are three classifiers in `dumbClassifiers.py`, one is implemented for you, the other two you will need to fill in appropriately.

The already implemented one is `AlwaysPredictOne`, a classifier that (as its name suggest) always predicts the positive class. We're going to use the `TennisData` dataset from `datasets.py` as a running example. So let's start up python and see how well this classifier does on this data. You should begin by importing `util`, `datasets`, `binary` and `dumbClassifiers`. Also, be sure you always have `from numpy import *` and `from pylab import *` activated.

```
>>> h = dumbClassifiers.AlwaysPredictOne({})
>>> h
AlwaysPredictOne
>>> h.train(datasets.TennisData.X, datasets.TennisData.Y)
>>> h.predictAll(datasets.TennisData.X)
array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
```

Indeed, it looks like it's always predicting one!

Now, let's compare these predictions to the truth. Here's a very clever way to compute accuracies (`REPORT1`: why is this computation equivalent to computing classification accuracy?):
```
>>> mean((datasets.TennisData.Y > 0) == (h.predictAll(datasets.TennisData.X) > 0))
0.6428571428571429
```
            
That's training accuracy; let's check test accuracy:
```
>>> mean((datasets.TennisData.Yte > 0) == (h.predictAll(datasets.TennisData.Xte) > 0))
0.5
```
            
Okay, so it does pretty badly. That's not surprising, it's really not learning anything!!! Now, let's use some of the built-in functionality to help do some of the grunt work for us. You'll need to import `runClassifier`.
```
>>> runClassifier.trainTestSet(h, datasets.TennisData)
Training accuracy 0.642857, test accuracy 0.5
```
            
Very convenient! Now, your first implementation task will be to implement the missing functionality in `AlwaysPredictMostFrequent`. This actually will "learn" something simple. Upon receiving training data, it will simply remember whether `+1` is more common or `-1` is more common. It will then always predict this label for future data. Once you've implemented this, you can test it:
```
>>> h = dumbClassifiers.AlwaysPredictMostFrequent({})
>>> runClassifier.trainTestSet(h, datasets.TennisData)
Training accuracy 0.642857, test accuracy 0.5
>>> h
AlwaysPredictMostFrequent(1)
```
            
Okay, so it does the same as `AlwaysPredictOne`, but that's because `+1` is more common in that training data. We can see a difference if we change to a different dataset: `CFTookAI` is a classification problem where we try to predict whether a student has taken AI based on the other classes they've taken.
```
>>> runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictOne({}), datasets.CFTookAI)
Training accuracy 0.515, test accuracy 0.42
>>> runClassifier.trainTestSet(dumbClassifiers.AlwaysPredictMostFrequent({}), datasets.CFTookAI)
Training accuracy 0.515, test accuracy 0.42
```
            
Since the majority class is "1", these do the same here. The last dumb classifier we'll implement is `FirstFeatureClassifier`. This actually does something slightly non-trivial. It looks at the first feature (i.e., X[0]) and uses this to make a prediction. Based on the training data, it figures out what is the most common class for the case when X[0] > 0 and the most common class for the case when X[0] <= 0. Upon receiving a test point, it checks the value of X[0] and returns the corresponding class. Once you've implemented this, you can check its performance:
```
>>> runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.TennisData)
Training accuracy 0.714286, test accuracy 0.666667
>>> runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.CFTookAI)
Training accuracy 0.515, test accuracy 0.42
>>> runClassifier.trainTestSet(dumbClassifiers.FirstFeatureClassifier({}), datasets.CFTookCG)
Training accuracy 0.545, test accuracy 0.49
```

(Here, `CFTookCG` is like `CFTookAI` but for computer graphics rather than artificial intelligence.) As we can see, this does better again on `TennisData`, but doesn't really help on AI.

---

### Vanilla Decision Trees (50%)

The main task is to implement a decision tree classifier. We provide a starting code `dt.py` that you should edit. In this implementation, the decision trees are in a form of simple data structures. Tree's each node has a `.isLeaf` field indicating if this node is a leaf (not an internal node). Leaf nodes have a `.label` field indicating what class to return at this leaf. Internal nodes have following fields: 

  * `.feature`: what feature to split on
  * `.left`: left child: what to do when the feature value is less than 0.5
  * `.right` right child: what to do when the feature value is at least 0.5. 
  
To see how the data structure works, please look at the `displayTree` function that prints out a tree.

First, implement the training function for the decision trees, Please start from the provided code. It should help you guard against corner cases. (Hint: `util.py` has some useful functions for implementing training.)

Once you've finished implementing the training function, you can test it on the `TennisData` data:
```
>>> h = dt.DT({'maxDepth': 1})
>>> h
Leaf 1

>>> h.train(datasets.TennisData.X, datasets.TennisData.Y)
>>> h
Branch 6
    Leaf 1.0
    Leaf -1.0
```

This is for a depth-1 decision tree (also known as a 'decision stump' as I mentioned in the class). If we make it deeper, we get something like:
```
>>> h = dt.DT({'maxDepth': 2})
>>> h.train(datasets.TennisData.X, datasets.TennisData.Y)
>>> h
Branch 6
    Branch 7
    Leaf 1.0
    Leaf 1.0
    Branch 1
    Leaf -1.0
    Leaf 1.0

>>> h = dt.DT({'maxDepth': 5})
>>> h.train(datasets.TennisData.X, datasets.TennisData.Y)
>>> h
Branch 6
    Branch 7
    Leaf 1.0
    Branch 2
        Leaf 1.0
        Leaf -1.0
    Branch 1
    Branch 7
        Branch 2
        Leaf -1.0
        Leaf 1.0
        Leaf -1.0
    Leaf 1.0
```
            
Now, let's implement inference function (`prediction`). You can test it by:
```
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.TennisData)
Training accuracy 0.714286, test accuracy 1
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 2}), datasets.TennisData)
Training accuracy 0.857143, test accuracy 1
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.TennisData)
Training accuracy 0.928571, test accuracy 1
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.TennisData)
Training accuracy 1, test accuracy 1
```
            
Now, let's see how well this does on our `CFTookCG` dataset:
```
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 1}), datasets.CFTookCG)
Training accuracy 0.56, test accuracy 0.48
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 3}), datasets.CFTookCG)
Training accuracy 0.6325, test accuracy 0.5
>>> runClassifier.trainTestSet(dt.DT({'maxDepth': 5}), datasets.CFTookCG)
Training accuracy 0.7475, test accuracy 0.6
```
            
The decision tree is better than the dumb classifiers! (on training data, as well as on test data) We hope that you can do even better in the future! We can use more `runClassifier` functions to plot learning curves and hyperparameter curves, for example:
```
>>> curve = runClassifier.learningCurveSet(dt.DT({'maxDepth': 5}), datasets.CFTookAI)
[snip]
>>> runClassifier.plotCurve('DT on AI', curve)
```

It plots accuracy of training and test as a function of the number of data points (in x-axis) used for training. 

`REPORT2`: We should see training accuracy (generally) decreasing and test accuracy (generally) increasing. Please dicsuss the reason that the training accuracy tend to decrease. Also, discuss the reason that the test accuracy not monotonically increasing. We can also plot similar curves by changing the maximum depth hyper-parameter:
```
>>> curve = runClassifier.hyperparamCurveSet(dt.DT({'maxDepth': 5}), 'maxDepth', [1,2,3,4,5,6,7,8,9,10], datasets.CFTookAI)
[snip]
>>> runClassifier.plotCurve('DT on AI (hyperparameter)', curve)
```

Now, the x-axis is the value of the maximum depth. 

`REPORT3`: If the implementation is correct, you will observe the training accuracy monotonically increasing and test accuracy tumbling. Please discuss the behavior of the training accuracy and test accuracy.

`REPORT4`: Train a decision tree on the CG dataset with a maximum depth of `3`. (Hint: in `datasets.CFTookCG.courseIds` and `.courseNames`, you'll find the corresponding course for each feature.) The first feature is a constant-one "bias" feature. In the write-up, first, draw out the decision tree for this classifier (but put in the actual course names/ids as the features). Then, discuss about this tree: do these courses are indicative of whether someone might take CG?

### Pruned Decision Trees (25%)

Now, let's implement the pruning function to reduce the size of the tree by the Chi-Square testing covered in the class.
```
>>> dt.pruneByChiSquare(0.05)
[snip]
```
`REPORT5`: Compare the test accuracy before and after the pruning by plotting a comparative graph. And discuss the effect of pruning in terms of training accuracy, test accuracy and generalization performance. Also, discuss the good hyperparameter (eg., 0.05) for each dataset.