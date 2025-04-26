import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
#import random as rd
#from matplotlib import pylab
import pickle
from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

os.chdir("C:/Users/olive/Documents/Git/ece547/ece547_project")
load_save = True
def printOut(a):
    b = pd.DataFrame(a)
    b.to_csv("./dataOutputTemp.csv")
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
    #training_answers.rename(columns={0 : 'is_malicious'}, inplace=True)

    for row in range(0, testing_answers.index.size):
        if (testing_answers.iat[row, 0] == "normal."):
            testing_answers.iat[row, 0] = 0
        else:
            testing_answers.iat[row, 0] = 1
    #testing_answers.rename(columns={0 : 'is_malicious'}, inplace=True)

    #use while debugging
    print(training_data)
    print(training_answers)
    print(testing_data)
    print(testing_answers)
    ####################

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
printOut(training_data)
print(training_answers)

scaler = StandardScaler()
training_data = pd.DataFrame(scaler.fit_transform(training_data))
testing_data = pd.DataFrame(scaler.transform(testing_data))
#####################

#nn = MLPClassifier(solver='sgd', alpha=0.0001, hidden_layer_sizes=(100,100), verbose=True, activation='relu', max_iter=1500, tol=1e-4, n_iter_no_change=20)#, learning_rate='adaptive')
#nn = nn.fit(training_data.to_numpy().astype(float), np.ravel(training_answers.to_numpy().astype(float)))
#predictions = nn.predict(testing_data.to_numpy().astype(float))
#scores = nn.predict_proba(testing_data.to_numpy().astype(float))
#
#print("Number of Epochs: ", nn.n_iter_)
#print("Final Mean Accuracy: ", nn.score(testing_data, testing_answers))
#print("Final Loss Achieved: ", nn.loss_)
#print("Lowest Loss Acheived: ", nn.best_loss_)
#print("Confusion Matrix:\n", confusion_matrix(testing_answers, predictions))
#print(" ")
#print(classification_report(testing_answers, predictions, target_names=['not malicious', 'malicious']))
#
#fig = plt.figure()
#plt.plot(nn.loss_curve_)
#plt.title("Loss Curve")
#plt.xlabel("# Epochs")
#plt.ylabel("Loss")
#plt.show()

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(51, 100)
        self.act = nn.ReLU()
        self.output = nn.Linear(100, 2)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# loss metric and optimizer
model = Multiclass()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# prepare model and training parameters
n_epochs = 200
batch_size = 5
batches_per_epoch = len(training_data) // batch_size

best_acc = - float(-999999.999)   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

training_answers = training_answers.astype(int)
testing_answers = testing_answers.astype(int)

# training loop
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    # set model in training mode and run through each batch
    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            X_batch = torch.tensor(training_data.to_numpy().astype(float), dtype=torch.float32)[start:start+batch_size]
            y_batch = torch.tensor(np.ravel(training_answers.to_numpy().astype(float)), dtype=torch.long)[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            bar.set_postfix(
                loss=float(loss),
                acc=float(acc)
            )
    # set model in evaluation mode and run through the test set
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(testing_data.to_numpy().astype(float), dtype=torch.float32)
        y_test = torch.tensor(testing_answers.to_numpy(), dtype=torch.long)

        y_pred = model(X_test)
        acc = (torch.argmax(y_pred, 1) == y_test).float().mean()
        print(f"Test Accuracy: {acc.item() * 100:.2f}%")
    

    y_pred = model(torch.tensor(testing_data.to_numpy().astype(float), dtype=torch.float32))
    ce = loss_fn(y_pred, torch.tensor(np.ravel(testing_answers.to_numpy().astype(float)), dtype=torch.long))
    acc = (torch.argmax(y_pred, 1) == torch.argmax(torch.tensor(np.ravel(testing_answers.to_numpy().astype(float)), dtype=torch.long), 0)).float().mean()
    ce = float(ce)
    acc = float(acc)
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(ce)
    test_acc_hist.append(acc)
    if acc > best_acc:
        best_acc = acc
        best_weights = copy.deepcopy(model.state_dict())
    print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

# Restore best model
model.load_state_dict(best_weights)

# Plot the loss and accuracy
plt.plot(train_loss_hist, label="train")
plt.plot(test_loss_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("cross entropy")
plt.legend()
plt.show()

plt.plot(train_acc_hist, label="train")
plt.plot(test_acc_hist, label="test")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()