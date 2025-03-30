import pandas as pd
#import os
#from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#import random as rd
#from matplotlib import pylab
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

load_save = True

if (not(load_save)):
    column_labels = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'udp', 
                        'tcp', 'other_protocol', 'private', 'smtp', 'ftp', 'ftp-data', 'telnet', 'domain_u', 'other_service',
                        'SF', 'REJ', 'other_flag']
    column_labels_classif = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
                            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
                            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
                            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'classification', 
                            'udp', 'tcp', 'other_protocol', 'private', 'smtp', 'ftp', 'ftp-data', 'telnet', 'domain_u', 'other_service',
                            'SF', 'REJ', 'other_flag']

    data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.data/kddcup.data", index_col=False, names=column_labels_classif, skiprows=lambda x: x % 100 != 0) #remove skiprows to process all
    training_data = data.copy(deep=True)
    training_data = training_data.drop(axis='columns', columns=['classification'])
    training_answers = data.copy(deep=True)
    training_answers = training_answers.drop(axis='columns', columns=column_labels)
    #testing_data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.testdata.unlabeled/kddcup.testdata.unlabeled", index_col=False, names=column_labels, skiprows=lambda x: x % 100 != 0) #remove skiprows to process all
    testing_data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.data/kddcup.data", index_col=False, names=column_labels_classif, skiprows=lambda x: x % 100 != 1) #remove skiprows to process all
    testing_answers = testing_data.copy(deep=True)
    testing_answers = testing_answers.drop(axis='columns', columns=column_labels)
    testing_data = testing_data.drop(axis='columns', columns=['classification'])

    for row in range(0, training_answers.index.size):
        if (training_answers.iat[row, 0] == "normal."):
            training_answers.iat[row, 0] = 0
        else:
            training_answers.iat[row, 0] = 1
    training_answers = training_answers.rename(columns={1:'is_malicious'})

    for row in range(0, testing_answers.index.size):
        if (testing_answers.iat[row, 0] == "normal."):
            testing_answers.iat[row, 0] = 0
        else:
            testing_answers.iat[row, 0] = 1
    testing_answers = testing_answers.rename(columns={1:'is_malicious'})

    #use while debugging
    print(training_data)
    print(training_answers)
    print(testing_data)
    print(testing_answers)
    ####################

    #training_data.append(axis='columns', columns=['udp', 'tcp', 'other_protocol'])
    #training_data.append(axis='columns', columns=['private', 'smtp', 'ftp' 'ftp-data', 'telnet', 'domain_u', 'other_service'])
    #training_data.append(axis='columns', columns=['SF', 'REJ', 'other_flag'])
    #
    #testing_data.append(axis='columns', columns=['udp', 'tcp', 'other_protocol'])
    #testing_data.append(axis='columns', columns=['private', 'smtp', 'ftp', 'ftp-data', 'telnet', 'domain_u','other_service'])
    #testing_data.append(axis='columns', columns=['SF', 'REJ', 'other_flag'])

    for row in range(0, training_data.index.size):
        match(training_data.iat[row,1]):
            case 'udp':
                training_data.iat[row, 41] = 1
                training_data.iat[row, 42] = 0
                training_data.iat[row, 43] = 0
            case 'tcp':
                training_data.iat[row, 41] = 0
                training_data.iat[row, 42] = 1
                training_data.iat[row, 43] = 0
            case _:
                training_data.iat[row, 41] = 0
                training_data.iat[row, 42] = 0
                training_data.iat[row, 43] = 1
        match(training_data.iat[row, 2]):
            case 'private':
                training_data.iat[row, 44] = 1
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 0
            case 'smtp':
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 1
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 0
            case 'ftp':
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 1
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 0
            case 'ftp-data':
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 1
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 0
            case 'telnet':
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 1
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 0
            case 'domain_u':
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 1
                training_data.iat[row, 50] = 0
            case _:
                training_data.iat[row, 44] = 0
                training_data.iat[row, 45] = 0
                training_data.iat[row, 46] = 0
                training_data.iat[row, 47] = 0
                training_data.iat[row, 48] = 0
                training_data.iat[row, 49] = 0
                training_data.iat[row, 50] = 1
        match(training_data.iat[row, 3]):
            case 'SF':
                training_data.iat[row, 51] = 1
                training_data.iat[row, 52] = 0
                training_data.iat[row, 53] = 0
            case 'REJ':
                training_data.iat[row, 51] = 0
                training_data.iat[row, 52] = 1
                training_data.iat[row, 53] = 0
            case _:
                training_data.iat[row, 51] = 0
                training_data.iat[row, 52] = 0
                training_data.iat[row, 53] = 1

    for row in range(0, testing_data.index.size):
        match(testing_data.iat[row,1]):
            case 'udp':
                testing_data.iat[row, 41] = 1
                testing_data.iat[row, 42] = 0
                testing_data.iat[row, 43] = 0
            case 'tcp':
                testing_data.iat[row, 41] = 0
                testing_data.iat[row, 42] = 1
                testing_data.iat[row, 43] = 0
            case _:
                testing_data.iat[row, 41] = 0
                testing_data.iat[row, 42] = 0
                testing_data.iat[row, 43] = 1
        match(testing_data.iat[row, 2]):
            case 'private':
                testing_data.iat[row, 44] = 1
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 0
            case 'smtp':
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 1
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 0
            case 'ftp':
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 1
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 0
            case 'ftp-data':
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 1
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 0
            case 'telnet':
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 1
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 0
            case 'domain_u':
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 1
                testing_data.iat[row, 50] = 0
            case _:
                testing_data.iat[row, 44] = 0
                testing_data.iat[row, 45] = 0
                testing_data.iat[row, 46] = 0
                testing_data.iat[row, 47] = 0
                testing_data.iat[row, 48] = 0
                testing_data.iat[row, 49] = 0
                testing_data.iat[row, 50] = 1
        match(testing_data.iat[row, 3]):
            case 'SF':
                testing_data.iat[row, 51] = 1
                testing_data.iat[row, 52] = 0
                testing_data.iat[row, 53] = 0
            case 'REJ':
                testing_data.iat[row, 51] = 0
                testing_data.iat[row, 52] = 1
                testing_data.iat[row, 53] = 0
            case _:
                testing_data.iat[row, 51] = 0
                testing_data.iat[row, 52] = 0
                testing_data.iat[row, 53] = 1

    training_data = training_data.drop(axis='columns', columns=['protocol_type', 'service', 'flag'])
    testing_data = testing_data.drop(axis='columns', columns=['protocol_type', 'service', 'flag'])
    
    with open('data_cache.pkl', 'wb') as outf:
        pickle.dump([training_data, training_answers, testing_data, testing_answers, data], outf) 
    print("pickled and saved!")    
else:
    with open('data_cache.pkl', 'rb') as inf: 
        [training_data, training_answers, testing_data, testing_answers, data] = pickle.load(inf) 
        
#####################
print(training_data)
print(testing_data)
#####################

nn = MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(100,100), verbose=True, activation='relu', max_iter=1500, tol=1e-4, n_iter_no_change=20)#, learning_rate='adaptive')
nn = nn.fit(np.asfarray(training_data), np.ravel(training_answers))
predictions = nn.predict(testing_data)
scores = nn.predict_proba(testing_data)

print("Number of Epochs: ", nn.n_iter_)
print("Final Mean Accuracy: ", nn.score(testing_data, testing_answers))
print("Final Loss Achieved: ", nn.loss_)
print("Lowest Loss Acheived: ", nn.best_loss_)
print("Confusion Matrix:\n", confusion_matrix(testing_data, predictions))
print(" ")
print(classification_report(testing_data, predictions, target_names=['not malicious', 'malicious']))

fig = plt.figure()
plt.plot(nn.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("# Epochs")
plt.ylabel("Loss")
plt.show()