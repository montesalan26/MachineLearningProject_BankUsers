# MachineLearningProject_BankUsers
In this project, I created different types of machine learning models to create an algorithm to predict wether users of bank will leave or stay as members of the bank, based off the dataset provided.

Necessary Libraries:
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from matplotlib import pyplot as plt  
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_auc_score  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier  
import seaborn as sns  
from sklearn.utils import shuffle, resample  
from imblearn.over_sampling import SMOTE  
from sklearn.preprocessing import StandardScaler  
from imblearn.under_sampling import RandomUnderSampler  
