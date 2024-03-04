class NeuralNetwork {
  weights
  biases
  leakyReLUAlpha = .1
  learnRate = .1
  clampRange = 10
  minError = .01
  maxEpochs = 3e3

  constructor (layers) {
    this.weights = this.initializeWeights(layers)
    this.biases = this.initializeBiases(layers)
  }

  train(trainData) {
    let startingWeights = JSON.stringify(this.weights)
    let startingBiases = JSON.stringify(this.biases)
    let startingError = this.getTotalError(trainData)
    let networkError = 0
    for (let i=0; i<this.maxEpochs; i++) {
      networkError = 0

      let changeNetworks = []
      for (let j=0; j<trainData.length; j++) {
        changeNetworks.push(this.makeChangeNetwork(trainData[j]))
      }
      this.applyChangeNetwork(this.averageChangeNetworks(changeNetworks), this.learnRate)

      networkError = this.getTotalError(trainData)

      if (networkError<this.minError) {
        return {epochs: i, msg: "finished early"}
      }

      //if (i%100==0) print("at epoch "+i+" the error is "+networkError)

    }
    if (startingError<networkError) {//revert changes if it did not improve
      this.weights = JSON.parse(startingWeights)
      this.biases = JSON.parse(startingBiases)
      return {epochs: this.maxEpochs, msg: "error is greater than when started(this has been reverted)"}
    }
    return {epochs: this.maxEpochs, msg: "ran out of epochs to get network error to min"}
  }

  getTotalError(trainData) {
    let totalDataError = 0
    for (let j=0; j<trainData.length; j++) {
      let qError = this.getError(this.run(trainData[j][0]), trainData[j][1])
      for (let k=0; k<qError.length; k++) {
        totalDataError += qError[k]**2
      }
    }
    return totalDataError
  }

  applyChangeNetwork(network, learnRate) {
    for (let i=0; i<this.weights.length; i++) {
      for (let j=0; j<this.weights[i].length; j++) {
        for (let k=0; k<this.weights[i][j].length; k++) {
          this.weights[i][j][k] += this.clampWeightChange(network.weights[i][j][k]*learnRate)
        }
      }
    }

    for (let i=0; i<this.biases.length; i++) {
      for (let j=0; j<this.biases[i].length; j++) {
       this.biases[i][j] += this.clampWeightChange(network.biases[i][j]*learnRate)
      }
    }
  }

  clampWeightChange(weight) {
    return Math.max(Math.min(weight, this.clampRange), -this.clampRange)
  }

  averageChangeNetworks(networks) {
    let weights = []

    for (let i=0; i<this.weights.length; i++) {
      weights[i] = []
      for (let j=0; j<this.weights[i].length; j++) {
        weights[i][j] = []
        for (let k=0; k<this.weights[i][j].length; k++) {
          weights[i][j][k] = 0
          for (let l=0; l<networks.length; l++) {
            weights[i][j][k] += networks[l].weights[i][j][k]
          }
          weights[i][j][k] /= networks.length
        }
      }
    }

    let biases = []

    for (let i=0; i<this.biases.length; i++) {
      biases[i] = []
      for (let j=0; j<this.biases[i].length; j++) {
        biases[i][j] = 0
        for (let k=0; k<networks.length; k++) {
          biases[i][j] += networks[k].biases[i][j]
        }
        biases[i][j] /= networks.length
      }
    }

    return {weights: weights, biases: biases}
  }

  makeChangeNetwork(trainingQuestion) {
    let modifiers = this.getBackwardsPassData(trainingQuestion)
    let predError = modifiers[0]
    let forwardPassData = modifiers[1]

    let weights = []
    for (let i=0; i<this.weights.length; i++) {
      weights[i] = []
      for (let j=0; j<this.weights[i].length; j++) {
        weights[i][j] = []
        for (let k=0; k<this.weights[i][j].length; k++) {
          weights[i][j][k] = predError[i][j] * forwardPassData[this.weights.length-i][k]
        }
      }
    }

    let biases = []
    for (let i=0; i<this.biases.length; i++) {
      biases[i] = []
      for (let j=0; j<this.biases[i].length; j++) {
        biases[i][j] = predError[i][j]
      }
    }

    return {weights: weights, biases: biases}
  }

  getBackwardsPassData(trainingQuestion) {
    let forwardPassData = this.forwardPass(trainingQuestion[0])
    let predError = [this.getError(forwardPassData[0], trainingQuestion[1])]

    for (let i=this.weights.length-1; i>=1; i--) {//going backwards through number of weight
      let layerError = []
      for (let j=0; j<this.weights[i][0].length; j++) {//going through weights(to)
        layerError[j] = 0
        for (let k=0; k<predError[0].length; k++) {//going through weights(from)
          layerError[j] += predError[0][k] * this.weights[i][k][j]//idr
        }
        layerError[j] *= this.leakyReLUDerivative(forwardPassData[i][j])//something something derivitive
      }
      predError.unshift(layerError)//add to front
    }

    return [predError, forwardPassData]
  }

  getError(pred, target) {
    let returnArr = []
    for (let i=0; i<pred.length; i++) {
      returnArr[i] = target[i] - pred[i]
    }
    return returnArr
  }

  run(input) {
    return this.forwardPass(input)[0]
  }

  forwardPass(input) {
    let output = [input]
    for (let i=0; i<this.weights.length; i++) {
      output.unshift(this.runPreceptron(output[0], this.weights[i], this.biases[i]))//adds to front
    }
    return output
  }

  runPreceptron(inputArr, weights2dArr, biasArr) {
    let output = []
    for (let i=0; i<weights2dArr.length; i++) {
      output[i] = 0
      for (let j=0; j<weights2dArr[0].length; j++) {
        output[i] += inputArr[j]*weights2dArr[i][j]
      }
      output[i] = this.leakyReLU(output[i]+biasArr[i])
      if (!isFinite(output[i])) output[i] = 0//handles nulls, infinities, and undefines
    }
    return output
  }

  leakyReLU(v) {
    return Math.max(v, v*this.leakyReLUAlpha)
  }

  leakyReLUDerivative(v) {
    if (v>0) {
      return 1
    } else {
      return this.leakyReLUAlpha
    }
  }

  initializeWeights(layers) {
    let weights = []
    for (let i=0; i<layers.length-1; i++) {
      weights.push(this.createRandomPreceptron(layers[i+1], layers[i]))
    }
    return weights
  }

  initializeBiases(layers) {
    let biases = []
    for (let i=0; i<layers.length-1; i++) {
      biases.push(this.createRandomArray(layers[i+1]))
    }
    return biases
  }

  createRandomPreceptron(l1, l2) {
    let preceptron = []
    for (let i=0; i<l1; i++) {
      preceptron.push(this.createRandomArray(l2))
    }
    return preceptron
  }

  createRandomArray(l) {
    let arr = []
    for (let i=0; i<l; i++) {
      arr[i] = (Math.random()-.5)*2
    }
    return arr
  }
}