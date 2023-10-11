# Cat Vision
###### A simple TensorFlow image classifier for cucumbers and snakes
![Example](https://github.com/henry9836/cat-vision/blob/master/imgs/inputs.gif?raw=true)

I built this image classifier to learn more about computer vision, it makes use of the TensorFlow library for Python to outperform some cats in identifying the difference between a cucumber and a snake.

### Setup
`$ pip install -r requirements.txt`

Download the dataset from Roboflow and organise it into data folders
```
project root
\ cat_vision
  \ test
   \ Cucumber
   | Snake
  \ train
   \ Cucumber
   | Snake
  \ valid
   \ Cucumber
   | Snake
```

### Usage
![Usage Image](https://raw.githubusercontent.com/henry9836/cat-vision/1.0/imgs/usage.png)

Simply run the application.

`$ ./catvision.py`

It will either generate a new model or use a saved one, once it is ready you will be presented with options.
#### Inputs
- `L (default)` This is the default option, it will load a random image from the testing directory
- `i` This will load a image from the internet via a url
#### Quitting
- `q` This will close the application without saving the model
- `s` This will save the model and then close the application

##### Roboflow dataset
https://universe.roboflow.com/nitroglycerin-films/cat-vision-0.5

### Supports URL image grabbing 
![Somethingfun](https://github.com/henry9836/cat-vision/blob/master/imgs/internet_images.gif?raw=true)

###### References
https://www.tensorflow.org/tutorials/images/classification

https://universe.roboflow.com/
