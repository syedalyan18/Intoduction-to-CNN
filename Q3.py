import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

(X_train,Y_train),(X_test,Y_test)=cifar10.load_data()

X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

Y_train=to_categorical(Y_train,10)
Y_test=to_categorical(Y_test,10)


datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


datagen.fit(X_train)

def create_model():

# Convolutional Layer 1
    model=models.Sequential()

    model.add(layers.Input(shape=(32,32,3)))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout((0.25)))


 # Convolutional Layer 2
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3),activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Dropout((0.25)))

    # Fully connected layers

    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout((0.5)))
    model.add(layers.Dense(10,activation='softmax'))

    return model

model=create_model()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(
    datagen.flow(X_train,Y_train,batch_size=64),
    epochs=20,
    validation_data=(X_test,Y_test),
    steps_per_epoch=X_train.shape[0] // 64
)

test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)

print(f"Test Accuracy : {test_accuracy}")

plt.plot(history.history['accuracy'],label="Training Accuracy")
plt.plot(history.history['val_accuracy'],label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.show()



plt.plot(history.history['loss'],label="Training loss")
plt.plot(history.history['val_loss'],label="Validation loss")
plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.show()