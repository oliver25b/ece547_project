import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import copy
#import random as rd
#from matplotlib import pylab
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torchinfo import summary


load_save = True
create_save = False
load_old_model = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir("C:/Users/olive/Documents/Git/ece547/ece547_project")
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

    data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.data/kddcup.data", index_col=False, names=column_labels_classif, skiprows=lambda x: x % 10 != 0) #remove skiprows to process all
    training_data = data.copy(deep=True)
    training_data = training_data.drop(axis='columns', columns=['classification'])
    training_answers = data.copy(deep=True)
    training_answers = training_answers.drop(axis='columns', columns=column_labels)
    testing_data = pd.read_csv("./kdd-cup-1999-data/versions/1/kddcup.data/kddcup.data", index_col=False, names=column_labels_classif, skiprows=lambda x: x % 4 != 1) #remove skiprows to process all
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
    
    if (create_save):
        with open('data_cache_new.pkl', 'wb') as outf:
            pickle.dump([training_data, training_answers, testing_data, testing_answers, data], outf) 
        print("pickled and saved!")    
else:
    with open('data_cache.pkl', 'rb') as inf: 
        [training_data, training_answers, testing_data, testing_answers, data] = pickle.load(inf) 

printOut(training_data)
        
#####################
#print("raw training data: ")
#print(training_data)
#print("raw training answers: ")
#print(training_answers)
#
#print("raw testing data: ")
#print(testing_data)
#print("raw testing answers: ")
#print(testing_answers)

scaler = StandardScaler()
training_data = pd.DataFrame(scaler.fit_transform(training_data))
testing_data = pd.DataFrame(scaler.transform(testing_data))
#####################
#print("training data after scaler: ")
#print(training_data)
#print("testing data after scaler: ")
#print(testing_data)

# Combine, shuffle together, split again
combined = pd.concat([training_data, training_answers], axis=1)
shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
training_data = shuffled.iloc[:, :-1]
training_answers = shuffled.iloc[:, -1]

#print("training data after combine and shuffle: ")
#print(training_data)
#print("training answers after combine and shuffle:  ")
#print(training_answers)

training_data_tensor = torch.tensor(training_data.to_numpy().astype(float), dtype=torch.float32, device=device)
training_answers_tensor = torch.tensor(np.ravel(training_answers.to_numpy().astype(float)), dtype=torch.float32, device=device)

testing_data_tensor = torch.tensor(testing_data.to_numpy().astype(float), dtype=torch.float32, device=device)
testing_answers_tensor = torch.tensor(np.ravel(testing_answers.to_numpy().astype(float)), dtype=torch.float32, device=device)

#print("training data tensor: ")
#print(training_data_tensor)
#print("training answers tensor: ")
#print(training_answers_tensor)
#print("testing data tensor: ")
#print(testing_data_tensor)
#print("testing answers tensor: ")
#print(testing_answers_tensor)

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(51, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.hidden1(x))
        x = self.act(self.hidden2(x))
        x = self.act(self.hidden3(x))
        x = self.output(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=5, threshold=1, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.threshold = threshold

    def __call__(self, score, model, path):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif (score < self.best_score) or (score >= self.threshold):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)
        if self.verbose:
            print('Validation score improved, saving model...')

# loss metric and optimizer, dropout
model = Multiclass().to(device)
early_stopping = EarlyStopping(patience=3, threshold=0.999, verbose=True)

# around 99% normal, 1% malicious
pos_weight = torch.tensor([99/1], device=device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# prepare model and training parameters
n_epochs = 80
batch_size = 512
batches_per_epoch = len(training_data) // batch_size
best_acc = - float('inf')   # init to negative infinity
best_weights = None
train_loss_hist = []
train_acc_hist = []
train_f1_hist = []
test_loss_hist = []
test_acc_hist = []
test_f1_hist = []
precision_hist = []
recall_hist = []
thresholds_hist = []
avg_precision_hist = []
best_thresholds_hist = []
best_f1_hist = []

if(not load_old_model):
    # training loop
    for epoch in range(n_epochs):
        # set model in training mode and run through each batch
        model.train()
        with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
            bar.set_description(f"Epoch {epoch}")
            for i in bar:
                # take a batch
                start = i * batch_size
                x_batch = training_data_tensor[start:start+batch_size].to(device)
                y_batch = training_answers_tensor[start:start+batch_size].to(device)

                # forward pass
                y_pred = model(x_batch)
                y_batch = y_batch.unsqueeze(1)
                loss = loss_fn(y_pred, y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                y_prob = torch.sigmoid(y_pred).float()

                # compute and store metrics
                y_pred_labels = (y_prob > 0.94).float()
                acc = (y_pred_labels.squeeze() == y_batch.float()).float().mean()
                f1 = f1_score(y_batch.cpu().numpy(), y_pred_labels.cpu().numpy())
                precision, recall, thresholds = precision_recall_curve(y_batch.cpu().numpy(), y_prob.detach().numpy())
                avg_precision = average_precision_score(y_batch.cpu().numpy(), y_prob.detach().numpy())

                train_loss_hist.append(float(loss))
                train_acc_hist.append(float(acc))
                train_f1_hist.append(float(f1))
                precision_hist.append(precision)
                recall_hist.append(recall)
                thresholds_hist.append(thresholds)
                avg_precision_hist.append(avg_precision)

                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        val_score = f1
        early_stopping(val_score, model, 'checkpoint.pth')

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break
    else:
        torch.load("best_model.pth")
        best_weights = copy.deepcopy(model.state_dict())
    # set model in evaluation mode and run through the test set
    model.eval()
    with torch.no_grad():
        x_test = testing_data_tensor.to(device)
        y_test = testing_answers_tensor.to(device)

        y_pred = model(x_test)
        y_prob = torch.sigmoid(y_pred).float()

        y_prob_flat = y_prob.view(-1).cpu().numpy()
        y_test_flat = y_test.view(-1).cpu().numpy().astype(int)

        y_pred_labels = (y_prob_flat > 0.94).astype(int)

        acc = (y_pred_labels == y_test_flat).mean()
        ce_loss = loss_fn(y_pred, y_test.unsqueeze(1))
        f1 = f1_score(y_test_flat, y_pred_labels)

        acc = float(acc)
        ce_loss = float(ce_loss)
        precision, recall, thresholds = precision_recall_curve(y_test_flat, y_prob_flat)
        #print(thresholds)
        avg_precision = average_precision_score(y_test.cpu().numpy(), y_prob.cpu().numpy())
        
        # Only keep every thousandth sample since there's over 80k per epoch
        thresholds_subsampled = thresholds[::1000]
        f1_scores_temp = np.array([
            f1_score(y_test_flat, (y_prob_flat >= t).astype(int))
            for t in thresholds_subsampled
        ])
         
        best_idx = np.argmax(f1_scores_temp)
        best_threshold = thresholds_subsampled[best_idx]
        best_f1 = f1_scores_temp[best_idx]

        print(f"Best threshold: {best_threshold:.4f}, F1 Score: {best_f1:.4f}")
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.', label=f'AP={avg_precision:.4f}')
        plt.scatter(recall[np.argmax(f1_scores_temp)], precision[np.argmax(f1_scores_temp)], color='red', label=f'Best Threshold={best_threshold:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Best Threshold')
        plt.grid(True)
        plt.legend()
        plt.show()

        best_f1_hist.append(best_f1)
        best_thresholds_hist.append(best_threshold)

        test_loss_hist.append(ce_loss)
        test_acc_hist.append(acc)
        test_f1_hist.append(f1)

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch} validation: Cross-entropy={ce_loss:.4f}, Accuracy={acc * 100:.2f}%, F1 Score={f1:.4f}")

if(not load_old_model):
    # Restore best model
    model.load_state_dict(best_weights)

    # Plot the loss and accuracy
    plt.plot(train_loss_hist, label="train")
    #plt.plot(np.arange(0, batches_per_epoch*n_epochs, batches_per_epoch), test_acc_hist, label="test")
    plt.xlabel("Learning Iteration")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()

    plt.plot(train_acc_hist, label="train")
    #plt.plot(np.arange(0, batches_per_epoch*n_epochs, batches_per_epoch), test_acc_hist, label="test")
    plt.xlabel("Learning Iteration")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(train_f1_hist, label="train")
    #plt.plot(np.arange(0, batches_per_epoch*n_epochs, batches_per_epoch), test_f1_hist, label="test")
    plt.xlabel("Learning Iteration")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.show()

    torch.save(best_weights, "best_model_01.pth")

summary(model, input_size=(batch_size, 51))