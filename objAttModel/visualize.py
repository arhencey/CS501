import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.math import confusion_matrix
from keras import backend as K
from skimage.transform import resize
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf

def plot_loss(history, filename='model_loss.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(filename)

def plot_accuracy(history, filename='model_accuracy.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename)

def plot_confusion(model, X_test, Y_test, labels, filename='confusion_matrix.png'):
    # print the frequencies of each answer
    print('Label frequencies:')
    (unique, counts) = np.unique(Y_test, return_counts=True)
    frequencies = np.asarray(counts)
    frequencies = np.reshape(frequencies, (frequencies.shape[0], 1))
    #print(frequencies)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    con_mat = confusion_matrix(labels=Y_test, predictions=y_pred).numpy()

    # normalize rows
    sum_of_rows = con_mat.sum(axis=1)
    normalized_con_mat = con_mat / sum_of_rows[:, np.newaxis]

    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()

    sns.heatmap(normalized_con_mat, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels, fmt='.2f')
    plt.tight_layout()
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)

def make_heatmap(model, video, filename='heatmap.png'):
    # clear the current firgure
    plt.clf()
    plt.cla()
    plt.close()

    prediction = model.predict(video)
    predicted_class = np.argmax(prediction)
    conv_layer = model.get_layer('conv3d_1')
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(video)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    firstFrame = heatmap.squeeze()[0]
    secondFrame = heatmap.squeeze()[1]

    upsample1 = resize(firstFrame, (64,64), preserve_range=True)
    upsample2 = resize(secondFrame, (64,64), preserve_range=True)
    vid = video[0] + 0.5

    fig = plt.figure(figsize=(7, 3))

    fig.add_subplot(1, 2, 1)
    plt.imshow(vid.squeeze()[0])
    plt.imshow(upsample1,alpha=0.5)

    fig.add_subplot(1, 2, 2)
    plt.imshow(vid.squeeze()[1])
    plt.imshow(upsample2,alpha=0.5)
    plt.savefig(filename)

def read_csv_log(filename='model_log.csv'):
    return pd.read_csv(filename, sep=',', engine='python')

def log_to_plots(filename='model_log.csv'):
    log = read_csv_log(filename)

    acc = log.iloc[:,1].values
    loss = log.iloc[:,2].values
    val_acc = log.iloc[:,3].values
    val_loss = log.iloc[:,4].values

    print(val_acc)

    # Plot loss
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('model_loss.png')

    # Plot accuracy
    plt.clf()
    plt.cla()
    plt.close()

    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_accuracy.png')
