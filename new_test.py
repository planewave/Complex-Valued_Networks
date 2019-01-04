from keras import layers
from keras import models
from complexnm.conv import ComplexConv2D

model = models.Sequential()
model.add(ComplexConv2D(32, (3, 3), activation='relu',
          input_shape=(28, 28, 2)))
print(model.summary())
