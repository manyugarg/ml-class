import numpy
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
config.epochs = 10

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
<<<<<<< HEAD
=======
print(X_train)
print(y_train)

print (X_train.shape)
#exit()
>>>>>>> 227c2671952ed49f2cbb22af23143025869b96e4

img_width = X_train.shape[1]
img_height = X_train.shape[2]
labels =["T-shirt/top","Trouser","Pullover","Dress",
    "Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

<<<<<<< HEAD
=======
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

>>>>>>> 227c2671952ed49f2cbb22af23143025869b96e4
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

<<<<<<< HEAD
=======

>>>>>>> 227c2671952ed49f2cbb22af23143025869b96e4
num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))
<<<<<<< HEAD
model.add(Dense(num_classes, activation='softmax'))
=======
model.add(Dense(num_classes))
>>>>>>> 227c2671952ed49f2cbb22af23143025869b96e4
model.compile(loss='mse', optimizer='adam',
                metrics=['accuracy'])

# Fit the model
<<<<<<< HEAD
model.fit(X_train[:10], y_train[:10], epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])


print(model.predict(X_train[:8]))
=======
model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
                    callbacks=[WandbCallback(data_type="image", labels=labels)])



>>>>>>> 227c2671952ed49f2cbb22af23143025869b96e4
