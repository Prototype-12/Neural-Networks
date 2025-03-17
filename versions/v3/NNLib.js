class NeuralNetwork {
  leakyReLUAlpha = .01
  learnRate = .1
  minError = .01
  maxEpochs = 3e3
  weights = []
  biases = []
  t = 0
  beta1 = .9
  beta2 = .999
  epsilon = 1e-8
  layers = []

  constructor (layers, options={}) {
    this.layers = layers
    this.weights = this.initWeights(layers)
    this.biases = this.initBiases(layers)
    this.resetAdam()
    this.mutate(options.weightsRandomness==undefined?2:options.weightsRandomness)
  }

  resetAdam() {
    this.t = 1
    this.mW = this.initWeights(this.layers)
    this.vW = structuredClone(this.mW)
    this.mB = this.initBiases(this.layers)
    this.vB = structuredClone(this.mB)
  }

  train(trainData) {
    let startingWeights = structuredClone(this.weights)
    let startingBiases = structuredClone(this.biases)
    let startingError = this.getTotalError(trainData)
    let networkError = 0
    this.resetAdam()
    for (let i=0; i<this.maxEpochs; i++) {

      let gradientNetworks = []
      for (let j=0; j<trainData.length; j++) {
        gradientNetworks.push(this.makeGradientNetwork(trainData[j]))
      }
      this.applyGradientNetworksAdam(gradientNetworks, this.learnRate)

      networkError = this.getTotalError(trainData)

      if (isNaN(networkError)) {//revert changes if NaN found
        this.weights = startingWeights
        this.biases = startingBiases
        return this.getExitMessage(i, 1, networkError)
      }

      if (networkError<this.minError) {
        return this.getExitMessage(i, 0, networkError)
      }

    }
    if (startingError<networkError) {//revert changes if it did not improve
      this.weights = startingWeights
      this.biases = startingBiases
      return this.getExitMessage(this.maxEpochs, 2, networkError)
    }
    return this.getExitMessage(this.maxEpochs, 3, networkError)
  }

  getExitMessage(epochs, exitCode, networkError) {
    let msg = ["Finished early", "NaN detected(maybe lower learnRate)(training changes revered)", "Score error is greater than when started(training changes revered)", "Ran out of epochs to get network error to min(maybe increase options.weightsRandomness or maxEpochs or minError)"][exitCode]
    return {epochs: epochs, exitCode: exitCode, msg: msg, networkError: networkError}
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

  applyGradientNetworksAdam(gradientNetworks, learnRate) {
    this.t++
    for (let i=0; i<this.weights.length; i++) {
      for (let j=0; j<this.weights[i].length; j++) {
        let dW = 0
        for (let k=0; k<gradientNetworks.length; k++) {
          dW += gradientNetworks[k].weights[i][j]
        }
        dW /= gradientNetworks.length
//        this.mW[i][j] = this.mW[i][j] * this.beta1 + (dW) * (1-this.beta1)
//        this.vW[i][j] = this.vW[i][j] * this.beta2 + (dW**2) * (1-this.beta2)
this.mW[i][j] += (dW-this.mW[i][j]) * (1-this.beta1)
this.vW[i][j] += ((dW**2)-this.vW[i][j]) * (1-this.beta2)
        let mWHat = this.mW[i][j] / (1-(this.beta1**this.t))
        let vWHat = this.vW[i][j] / (1-(this.beta2**this.t))
        this.weights[i][j] += learnRate * mWHat / ((vWHat**.5) + this.epsilon)
      }
    }
    for (let i=0; i<this.biases.length; i++) {
      for (let j=0; j<this.biases[i].length; j++) {
        let dB = 0
        for (let k=0; k<gradientNetworks.length; k++) {
          dB += gradientNetworks[k].biases[i][j]
        }
        dB /= gradientNetworks.length
//        this.mB[i][j] = this.mB[i][j] * this.beta1 + (dB) * (1-this.beta1)
//        this.vB[i][j] = this.vB[i][j] * this.beta2 + (dB**2) * (1-this.beta2)
this.mB[i][j] += (dB-this.mB[i][j]) * (1-this.beta1)
this.vB[i][j] += ((dB**2)-this.vB[i][j]) * (1-this.beta2)
        let mBHat = this.mB[i][j] / (1-(this.beta1**this.t))
        let vBHat = this.vB[i][j] / (1-(this.beta2**this.t))
        this.biases[i][j] += learnRate * mBHat / ((vBHat**.5) + this.epsilon)
      }
    }
  }

  makeGradientNetwork(trainingQuestion) {
    let modifiers = this.getBackwardsPassData(trainingQuestion)
    let predError = modifiers[0]
    let forwardPassData = modifiers[1]

    let weights = []
    for (let i=0; i<this.weights.length; i++) {
      weights[i] = []
      for (let j=0; j<this.weights[i].length; j++) {
        weights[i][j] = predError[i][j%this.biases[i].length] * forwardPassData[this.weights.length-i][Math.floor(j/this.biases[i].length)]
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

    for (let i=this.weights.length-1; i>=1; i--) {//black magic
      let layerError = []
      let l = this.biases[i-1].length//this.weights[i].length/predError[0].length
      for (let j=0; j<this.weights[i].length; j++) {
        let a = j%l
        let b = Math.floor(j/l)
        if (b==0) layerError[a] = 0
        layerError[a] += predError[0][b] * this.weights[i][j]
        if (b==l-1) layerError[a] *= this.leakyReLUDerivative(forwardPassData[i][a])
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
      output.unshift(this.forwardPassLayer(output[0], this.weights[i], this.biases[i]))//adds to front
    }
    return output
  }

  forwardPassLayer(inputArr, weightsArr, biasArr) {
    let output = []
    for (let i=0; i<weightsArr.length; i++) {
      let a = i%inputArr.length
      let b = Math.floor(i/inputArr.length)
      if (a==0) output[b] = 0
      output[b] += inputArr[a] * weightsArr[i]
      if (a==inputArr.length - 1) output[b] = this.leakyReLU(output[b] + biasArr[b])
    }
    return output
  }

  leakyReLU(v) {
    return Math.max(v, v*this.leakyReLUAlpha)
  }

  leakyReLUDerivative(v) {
    return v>0?1:this.leakyReLUAlpha
  }

  initWeights(layers) {
    let weights = []
    for (let i=0; i<layers.length-1; i++) {
      weights.push(new Array(layers[i]*layers[i+1]).fill(0))
    }
    return weights
  }

  initBiases(layers) {
    let biases = []
    for (let i=0; i<layers.length-1; i++) {
      biases.push(new Array(layers[i+1]).fill(0))
    }
    return biases
  }

  mutate(mutationStrength) {
    for (let i=0; i<this.weights.length; i++) {
      this.weights[i] = this.mutateArray(this.weights[i], mutationStrength)
    }
    for (let i=0; i<this.biases.length; i++) {
      this.biases[i] = this.mutateArray(this.biases[i], mutationStrength)
    }
  }

  mutateArray(arr, mutationStrength) {
    for (let i=0; i<arr.length; i++) arr[i] += (Math.random()-.5)*mutationStrength
    return arr
  }

  clone() {
    let clone = new NeuralNetwork([1])
    clone.layers = structuredClone(this.layers)
    clone.weights = structuredClone(this.weights)
    clone.biases = structuredClone(this.biases)
    clone.leakyReLUAlpha = this.leakyReLUAlpha
    clone.learnRate = this.learnRate
    clone.minError = this.minError
    clone.maxEpochs = this.maxEpochs
    return clone
  }
}
