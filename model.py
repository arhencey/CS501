from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv3D, MaxPooling3D, Flatten, Multiply
from tensorflow.keras.optimizers import Adam

def build_model(vid_shape, vocab_size, num_answers, big_model):
  # The CNN
  vid_input = Input(shape=vid_shape)
  x1 = Conv3D(8, (3,3,1), padding='same')(vid_input)
  x1 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x1)
  x1 = Conv3D(16, (3,3,1), padding='same')(x1)
  x1 = MaxPooling3D(pool_size=(2,2,1), padding='same')(x1)
  if big_model:
    x1 = Conv3D(32, 3, padding='same')(x1)
    x1 = MaxPooling3D(padding='same')(x1)
  x1 = Flatten()(x1)
  x1 = Dense(32, activation='tanh')(x1)

  # The question network
  q_input = Input(shape=(vocab_size,))
  x2 = Dense(32, activation='tanh')(q_input)
  x2 = Dense(32, activation='tanh')(x2)

  # Merge -> output
  out = Multiply()([x1, x2])
  out = Dense(32, activation='tanh')(out)
  out = Dense(num_answers, activation='softmax')(out)

  model = Model(inputs=[vid_input, q_input], outputs=out)
  model.compile(Adam(lr=10e-6), loss='categorical_crossentropy', metrics=['accuracy'])

  return model
