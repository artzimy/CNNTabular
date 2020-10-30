#!/usr/bin/env python
# coding: utf-8

# # Test Prediction

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from collections import Counter

from Macros import *
from TabConvData import *





def predict_test(path, csv_name, learner, majority_of_votes, show_preds=True):
    test_data_list = create_set(path, csv_name)
    test_data, test_labels = test_data_list.train.x, test_data_list.train.y
    classes = ['Controls', 'Ependymoma', 'Medulloblastoma', 'PilocyticAstrocytoma']
    
    if (majority_of_votes):
        true_labels = compress_labels(test_data, test_labels, classes)
        predictions = predict_set_mov(test_data, learner, classes)
        predicted_labels = [elem[0] for elem in predictions]
    else:
        true_labels = [classes[int(label)] for label in test_labels]
        predicted_labels = predict_set(test_data, learner, classes)
        
    if show_preds:
        if (majority_of_votes):
            print(predictions)
        else:
            print(predicted_labels)

    acc = calc_accuracy(predicted_labels, true_labels)
    print("Test Accuracy = "+str(acc))
    return true_labels, predicted_labels


# **Prediction Functions**

# In[ ]:


def predict(item, learner, classes):
    cont = item.data[1].unsqueeze(dim=0).cuda()
    img = item.data[2].unsqueeze(dim=0).cuda()
    probs = learner.model(cont, img)[0]
    max_val, max_ind = probs.max(0)
    return classes[max_ind]
    
def predict_patient(patient_list, learner, classes):
    n = len(patient_list)
    preds = []
    for i in range(n):
        item = patient_list[i]
        preds.append(predict(item, learner, classes))
    pred, score = majority(preds)
    return pred, score/n

def predict_set(items, learner, classes):
    preds = [predict(item, learner, classes) for item in items]
    return preds

def predict_set_mov(items, learner, classes):
    i = 0
    n = len(items)
    patients = []
    while (i < n):
        patient = []
        pid = items[i].pid
        while (i < n):
            patient_item = items[i] 
            cur_pid = patient_item.pid
            if (cur_pid != pid): break
            patient.append(patient_item)
            i += 1
        patients.append(patient)
        
    preds = []
    for patient in patients:
        preds.append(predict_patient(patient, learner, classes))
        
    return preds  


# **Auxiliary Functions**

# In[ ]:


def create_set(path, csv_name):
    df = pd.read_csv(path + '/' + csv_name)
    df[PATIENT] = df[ID].apply(lambda image_name: image_name.split('_')[1])
    df[ORIG_AGE] = df[AGE]
    if CLASS in df.columns:
        data_list = (TabConvList.from_df(df, cont_names=cont_names, procs=[Normalize],
                     path=path, imgs=[ID],suffix='.png')
                     .split_none().label_from_df(cols=CLASS, label_cls=CategoryList)
                     .transform(size=256, resize_method=ResizeMethod.SQUISH, padding_mode='zeros'))
    else:
        data_list = (TabConvList.from_df(df, cont_names=cont_names, procs=[Normalize],
                     path=path, imgs=[ID],suffix='.png')
                     .transform(size=256, resize_method=ResizeMethod.SQUISH, padding_mode='zeros'))

    return data_list

def compress_labels(data, labels, classes):
    i = 0
    n = len(data)
    labels_by_patient = []
    pid = ''
    for i in range(n):
        cur_pid = data[i].pid
        if cur_pid != pid:
            labels_by_patient.append(classes[int(labels[i])])
            pid = cur_pid
    
    return labels_by_patient


# **Results**

# In[ ]:


def majority(lst):
    c = Counter(lst)
    max_key = max(c, key=c.get)
    max_value = c[max_key]
    return max_key, max_value

def calc_accuracy(predicted_labels, true_labels):
    n = len(predicted_labels)
    correct_labels = 0
    for i in range(n):
        if predicted_labels[i] == true_labels[i]:
            correct_labels += 1
    accuracy = correct_labels / n
    return accuracy

# print predictions next to true labels
def show_predictions(predictions, true_labels):
    n = len(true_labels)
    print("*True Label*        *Predicted Label*          *Confidence Level*")
    for i in range(n):
        print(true_labels[i] +"         "+ predictions[i][0] +"         "+ str(predictions[i][1]))


# # Test Explainability

# **Interface**
# 
# Given a TabConv learner, load test data and predict on its items.
# 
# Show predictions of combined items (MRI image and patient age) in test[from_idx,to_idx] - 
# provide appropriate range consist up to 25 consecutive items.

# In[ ]:


def show_test_results(path, csv, learner, from_idx=0, to_idx=24):
    classes = ['Controls', 'Ependymoma', 'Medulloblastoma', 'PilocyticAstrocytoma']
    
    test_data_list = create_set(path, csv)    
    test_data, test_labels = test_data_list.train.x, test_data_list.train.y
    
    n = len(test_data)
    check_indexes_validity(n, from_idx, to_idx)

    predicted_labels = predict_set(test_data, learner, classes)
    show_test_xyzs(test_data, test_labels, predicted_labels, from_idx, to_idx)


# **Auxiliary Functions**

# In[1]:


def check_indexes_validity(n, from_idx, to_idx):
    assert (from_idx <= to_idx and to_idx - from_idx + 1 <= 25 and
        from_idx >= 0 and to_idx >= 0 and from_idx <= n-1 and to_idx <= n-1), "Unvalid indexes"
    
def show_result(axs, img, age, actual, prediction, fontsize):
    axs.imshow(img.permute(1,2,0))
    axs.set_title(''.join([str(actual), ' ; ', str(age), '\n', prediction]), fontsize=fontsize)
    
def show_test_xyzs(xs, ys, zs, from_idx, to_idx, figsize:(int,int)=(90,100), **kwargs):
    "Show the `xs` and `ys` and 'zs' (predictions) on a figure of `figsize`. `kwargs` are passed to the show method."
    num_results = to_idx - from_idx + 1
    rows = int(np.ceil(math.sqrt(num_results)))
    adj_figsize = (5*rows, 5*rows)

    age = [item.orig_age for item in xs]
    img = torch.cat([item.img.data.unsqueeze(0) for item in xs], 0)
    age = age[from_idx:to_idx+1]
    img = img[from_idx:to_idx+1]
    
    fig, axs = plt.subplots(rows, rows, figsize=adj_figsize, constrained_layout=True)
    fig.suptitle("Actual ; Age(months)" + "\n" + "Prediction" + "\n", fontsize=adj_figsize[0]+10, weight='bold')
    
    if rows == 1:
        show_result(axs, img[0], age[0], ys[0], zs[0], 15)
        axs.axis('off')
    else:
        axs = axs.flatten()
        for i in range(num_results):
            show_result(axs[i], img[i], age[i], ys[i], zs[i], adj_figsize[0])
        
        for i in range(rows*rows):
            axs[i].axis('off')

