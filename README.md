# Neural-Networks

Simple library for neural networks(includes backpropagation)

## How to install

Import from Github

```html
<script src="https://commandninja.github.io/Neural-Networks/neuralNetworkLib.js"></script>
```

or import from file

```html
<script src="neuralNetworkLib.js"></script>
```

## Initializing

Initializing creates a random neural network(random weights and biases from -1 to 1)

And it supports creating multiple hidden layers

```js
var ai = new Neuralnetwork([2, 3, 1])
var ai2 = new Neuralnetwork([5, 10, 10, 1])
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


[An array containing [arrays that contain an [inputs] and an [outputs array]]]

## Changing settings

Settings are local to a network

```js
ai.leakyReLUAlpha = .01 // messes with the leakyReLU function
ai.learnRate = .01 // multipler for the changes of each epoch
ai.clampRange = 1 // changes the max, a weight or bias is allowed to change each epoch
ai.minError = .001 // min error needed to auto finish
ai.maxEpochs = 5000 // max tries a network will try to get better
```

## Demo

[DEMO](https://commandninja.github.io/Neural-Networks)

## Advanced

```js
ai.getTotalError(trainingData) // this gets the total error((target-pred)**2)
ai.forwardPass(inputArr) // this will return the values of each layer as the input passes through it
```

## Notes

Uses leakyReLU

Not fast (I should have used matrixes)

When training data is input its trained as batches

Any non finite number(null, undefined, Infinity, -Infinity) will be turned to 0

The trainer will return the original network if the current error exceeds the original error