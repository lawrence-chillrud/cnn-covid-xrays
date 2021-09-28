# Author: Lawrence Chillrud
# UNI: lgc2139
# Date: 4/28/20
# File: explore.py for Nakul Verma's COMS 4771 Kaggle Competition

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
from PIL import Image
from os import listdir
from scipy.stats import gaussian_kde
import seaborn as sns


def load_data():
    train_df = pd.read_csv('4771-sp20-covid/train.csv')
    test_df = pd.read_csv('4771-sp20-covid/test.csv')

    train_id = train_df['id'].values
    train_filename = train_df['filename'].values
    train_label = train_df['label'].values

    test_id = test_df['id'].values
    test_filename = test_df['filename'].values

    """
    list_ds = tf.data.Dataset.list_files('4771-sp20-covid/train/*')
    for f in list_ds.take(5):
          print(f.numpy())
    """
    return (train_id, train_filename, train_label), (test_id, test_filename)

# [covid, viral, bacterial, normal]
def pixel_jointplot(dataset='train'):
    wd = '4771-sp20-covid/%s/' % (dataset)
    files = [f for f in listdir(wd)]
    x_shape = []
    y_shape = []
    shapes = []
    for im_p in files:
        im = np.array(Image.open(wd+im_p))
        x_shape.append(im.shape[0])
        y_shape.append(im.shape[1])
     
    x_shape = np.array(x_shape) # width 
    y_shape = np.array(y_shape) # height
    df = pd.DataFrame({'Image Width (Pixels)':x_shape, 'Image Height (Pixels)':y_shape})
    p = sns.jointplot(x="Image Width (Pixels)", y="Image Height (Pixels)", data=df, kind="kde");
    p.ax_joint.set_xticks(np.arange(0,3250,250))
    p.ax_joint.set_yticks(np.arange(0,3250,250))
    plt.show()

def pixel_scatterplot(dataset='train'):
    wd = '4771-sp20-covid/%s/' % (dataset)
    files = [f for f in listdir(wd)]
    x_shape = []
    y_shape = []
    shapes = []
    for im_p in files:
        im = np.array(Image.open(wd+im_p))
        x_shape.append(im.shape[0])
        y_shape.append(im.shape[1])
     
    x_shape = np.array(x_shape) # width 
    y_shape = np.array(y_shape) # height
    xy = np.vstack([x_shape,y_shape])
    z = gaussian_kde(xy)(xy)
    index = z.argsort()
    x, y, z = x_shape[index], y_shape[index], z[index]
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50, edgecolor='')
    plt.title('Distribution of %s Dataset Image Sizes' % (dataset), y = 1)
    plt.xlabel('Pixel Width')
    plt.ylabel('Pixel Height')
    plt.show()

def breakdown_bar(labels):
    counts = np.zeros(4)
    covid_ids = list()
    viral_ids = list()
    bac_ids = list()
    normal_ids = list()

    key = ['covid', 'viral', 'bacterial', 'normal']
    for idx, i in enumerate(labels):
        if i == 'covid':
            counts[0] += 1
            covid_ids.append(idx)
        elif i == 'viral':
            counts[1] += 1
            viral_ids.append(idx)
        elif i == 'bacterial':
            counts[2] += 1
            bac_ids.append(idx)
        else:
            counts[3] += 1
            normal_ids.append(idx)
    
    print(covid_ids)
    print(len(covid_ids))
    def autolabel(rects):
        # Attach a text label above each bar displaying its height
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                    '%d' % int(height), ha='center', va='bottom')
    
    fig, ax = plt.subplots()
    rect1 = ax.bar(key, counts, align='center', alpha=0.5)
    x_pos = np.arange(len(key))
    ax.set_xticks(x_pos, key)
    ax.set_ylabel('Raw counts')
    plt.title('Training Dataset Breakdown by Class', y=1)
    autolabel(rect1)
    plt.show()

def pixel_pdf(dataset='train'):
    wd = '4771-sp20-covid/%s/' % (dataset)
    files = [f for f in listdir(wd)]
    x_shape = []
    y_shape = []
    for im_p in files:
        im = np.array(Image.open(wd+im_p))
        x_shape.append(im.shape[0])
        y_shape.append(im.shape[1])
     
    x_shape = np.array(x_shape) # width 
    y_shape = np.array(y_shape) # height
    # Density Plot
    sns.distplot(x_shape, hist=False, kde=True, kde_kws={'linewidth': 4}, label='width')
    sns.distplot(y_shape, hist=False, kde=True, kde_kws={'linewidth': 4}, label='height')
    plt.title('%s Dataset Image Sizes' % (dataset))
    plt.xlabel('Pixels')
    plt.show()

def print_ims(training_ims, training_labels, num_printed=24):
    wd = '4771-sp20-covid/'
    plt.figure(figsize=(10,10))
    for i in range(num_printed):
        plt.subplot(4,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(Image.open(wd + training_ims[i]).resize((512,512)), cmap=plt.cm.binary)
        plt.xlabel(training_labels[i])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    (train_id, train_filename, train_label), (test_id, test_filename) = load_data()
    print_ims(train_filename, train_label)
    #breakdown_bar(train_label)
    #pixel_jointplot()
