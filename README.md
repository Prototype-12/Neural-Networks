# Neural-Networks

Simple library for neural networks

## How to install

Import from Github

```html
<script src="https://prototype-12.github.io/Neural-Networks/versions/v3/NNLib.js"></script>
```

or download it and import from the file

```html
<script src="NNLib.js"></script>
```

## Initializing

Initializing creates a random neural network(random weights and biases from -1 to 1)

And it supports creating multiple hidden layers

```js
var ai = new Neuralnetwork([2, 3, 1])
var ai2 = new Neuralnetwork([5, 10, 10, 5])
```

## Running

```js
ai.run([3, 4])
```

This will return an array with a length of 1 as the first declaration specified

## Training

Training data is formulated like.

```js
var trainingData = [
[[0,0],[0]],
[[0,1],[1]],
[[1,0],[1]],
[[1,1],[0]]
]

ai.train(trainingData) //this will train it
```


Training data is an [array containing [arrays that contain an [inputs] and an [outputs array]]]

## Changing settings

Settings are local to a network

```js
ai.leakyReLUAlpha = .01 // messes with the leakyReLU function
ai.learnRate = .01 // multipler for the changes of each epoch
ai.minError = .001 // min error needed to auto finish
ai.maxEpochs = 5000 // max tries a network will try to get better
```

## Demo

[DEMO](https://prototype-12.github.io/Neural-Networks)

## Advanced

```js
ai.getTotalError(trainingData) // this gets the total error((target-pred)**2)
ai.forwardPass(inputArr) // this will return the values of each layer as the input passes through it
```

All functions and values in the class are public so they can be changed

## Notes

Uses leakyReLU

v3 Uses Adam

When training data is input, it's trained as a batch

The trainer will return the original network if the current error exceeds the original error