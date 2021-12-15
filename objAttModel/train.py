from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import argparse
from model import build_model
from prepare_data import setup
from visualize import plot_loss, plot_accuracy, plot_confusion, make_heatmap
from tensorflow.math import confusion_matrix

from PIL import Image, ImageSequence
import numpy as np
import math

# Support command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()

if args.use_data_dir:
    print('Using data directory')

# Prepare data
train_X_first_objects, train_X_second_objects, train_X_seqs, train_Y, test_X_first_objects, test_X_second_objects, test_X_seqs, test_Y, vid_shape, vocab_size, num_answers, all_answers, _, _ = setup(args.use_data_dir)

print('\n--- Building model...')
model = build_model(vid_shape, vocab_size, num_answers)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
csv_logger = CSVLogger('model_log.csv', append=False) # set append=True if continuing training

print('\n--- Training model...')
history = model.fit(
        [train_X_first_objects, train_X_second_objects, train_X_seqs],
        train_Y,
        validation_data=([test_X_first_objects, test_X_second_objects, test_X_seqs], test_Y),
        shuffle=True,
        epochs=500,
        callbacks=[checkpoint, es, csv_logger],
)

print('\n--- Generating plots...')
plot_loss(history)
plot_accuracy(history)

test_Y = np.argmax(test_Y, axis=1)
plot_confusion(model, [test_X_first_objects, test_X_second_objects, test_X_seqs], test_Y, all_answers)
