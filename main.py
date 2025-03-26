import pandas as pd
#import os
#from pathlib import Path
#import matplotlib.pyplot as plt
import numpy as np
import random as rd
#from matplotlib import pylab
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report

column_labels = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
column_labels_classif = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'classification']

data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.data/kddcup.data", index_col=False, names=column_labels_classif)
print(data)
training_data = data.copy(deep=True)
training_data = training_data.drop(axis='columns', columns=['classification'])
print(training_data)
training_answers = data.copy(deep=True)
training_answers = training_answers.drop(axis='columns', columns=column_labels)
print(training_answers)
testing_data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.testdata.unlabeled/kddcup.testdata.unlabeled", index_col=False, names=column_labels)
print(testing_data)

#for row in range(0, training_data.index.size):
 #   if (training_data[classification] == "normal."):
        