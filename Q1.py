import matplotlib.pyplot as plt
from torchvision import datasets,transforms

transform=transforms.ToTensor()
train_dataset=datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)

fig,axes=plt.subplots(1,5,figsize=(3,3))

for i in range(5):
    image,label=train_dataset[i]
    axes[i].imshow(image.permute(1,2,0))
    axes[i].axis('off')
    axes[i].set_title(f"Label : {label}")
plt.show()

# image,label=train_dataset[0]
# print(f"label : {label}")
# print(f"Image Shape : {image.shape}")
# print("Image Values :")
# print(image)


import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D

model=tf.keras.Sequential([
     Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)),
     MaxPooling2D((2,2)),
     Flatten(),
     Dense(128,activation="relu"),
     Dense(10,activation="softmax")
])

model.compile(optimizer='adam',loss='sparse_categorical_crosssentropy',metrics=['accuracy'])

print("Tensorflow CNN Model is ready !!")


import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__()
        self.conv1=nn.Conv2d(3,32,kernel_size=3,activation='relu')
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(32 * 12 * 15 * 128)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=x.view(-1,32 * 15 * 15)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)

print("Pytorch CNN Model is ready !!")