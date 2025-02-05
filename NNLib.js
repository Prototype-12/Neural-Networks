class NeuralNetwork {
  leakyReLUAlpha = .01
  learnRate = .01
  minError = .01
  maxEpochs = 1e3
  weights = []
  biases = []

  constructor (layers, randomStrength=20) {
      this.initializeWeights(layers)//layer format
      this.initializeBiases(layers)
      this.mutate(randomStrength)
  }

  train(trainData) {
    let startingWeights = structuredClone(this.weights)
    let startingBiases = structuredClone(this.biases)
    let startingError = this.getTotalError(trainData)
    let networkError = 0

    for (let i=0; i<this.maxEpochs; i++) {

      let changeNetworks = []
      for (let j=0; j<trainData.length; j++) {
        changeNetworks.push(this.makeChangeNetwork(trainData[j]))
      }
      //this.applyChangeNetwork(this.averageChangeNetworks(changeNetworks), this.learnRate)//this.learnRate*(i/this.maxEpochs))
      this.applyChangeNetworks(changeNetworks, this.learnRate)//this.learnRate*(i/this.maxEpochs))

      networkError = this.getTotalError(trainData)

      if (isNaN(networkError)) {
        this.weights = startingWeights
        this.biases = startingBiases
        return this.getExitMessage(i, 1, networkError)
      }

      if (networkError<this.minError) {
        return this.getExitMessage(i, 0, networkError)
      }

      //if (i%1e3==0) console.log("at epoch "+i+" the error is "+networkError)

    }
    if (startingError<networkError) {//revert changes if it did not improve
      this.weights = startingWeights
      this.biases = startingBiases
      return this.getExitMessage(this.maxEpochs, 2, networkError)
    }
    return this.getExitMessage(this.maxEpochs, 3, networkError)
  }

  getExitMessage(epochs, exitCode, networkError) {
    let msg = ["Finished early", "NaN detected(maybe lower learnRate)(training changes revered)", "Score error is greater than when started(training changes revered)", "Ran out of epochs to get network error to min"][exitCode]
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

  applyChangeNetworks(networks, learnRate) {
    let multiplier = learnRate/networks.length
    for (let k=0; k<networks.length; k++) {
      for (let i=0; i<this.weights.length; i++) {
        for (let j=0; j<this.weights[i].length; j++) {
          this.weights[i][j] += networks[k].weights[i][j]*multiplier
        }
      }
      for (let i=0; i<this.biases.length; i++) {
        for (let j=0; j<this.biases[i].length; j++) {
          this.biases[i][j] += networks[k].biases[i][j]*multiplier
        }
      }
    }
  }

  makeChangeNetwork(trainingQuestion) {
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

    for (let i=this.weights.length-1; i>=1; i--) {
      let layerError = []
      let l = this.biases[i-1].length//this.weights[i].length/predError[0].length
      for (let j=0; j<this.weights[i].length; j++) {
        let a = j%l
        let b = Math.floor(j/l)
        if (b==0) layerError[a] = 0//,console.log("0 layer")
        layerError[a] += predError[0][b] * this.weights[i][j]//,console.log(a,b,l)
        if (b==l-1) layerError[a] *= this.leakyReLUDerivative(forwardPassData[i][a])//,console.log("layer finish")
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
      output[b] += inputArr[a] * weightsArr[i]//;console.log(a, b)
      if (a==inputArr.length - 1) output[b] = this.leakyReLU(output[b] + biasArr[b])//;console.log("layer finish")
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

  initializeWeights(layers, randomStrength) {
    for (let i=0; i<layers.length-1; i++) {
      this.weights.push(new Array(layers[i]*layers[i+1]).fill(0))
      //weights.push(this.createRandomArray(layers[i]*layers[i+1], randomStrength))
    }
  }

  initializeBiases(layers, randomStrength) {
    for (let i=0; i<layers.length-1; i++) {
      this.biases.push(new Array(layers[i+1]).fill(0))
      //biases.push(this.createRandomArray(layers[i+1], randomStrength))
    }
  }

  //createRandomArray(l, randomStrength) {
  //  return new Array(l).fill(0)
/*
    let arr = new Array(l)//[]
    for (let i=0; i<l; i++) {
      arr[i] = (Math.random()-.5)*randomStrength
    }
    return arr
*/
  //}

  mutate(mutationStrength) {
    for (let i=0; i<this.weights.length; i++) {
      this.weights[i] = this.mutateArray(this.weights[i], mutationStrength)
    }
    for (let i=0; i<this.biases.length; i++) {
      this.biases[i] = this.mutateArray(this.biases[i], mutationStrength)
    }
/*
    if (Math.random()<.5) {
      let weightLayer = this.weights[Math.floor(Math.random()*this.weights.length)]
      weightLayer = this.mutateArray(weightLayer, mutationStrength)
    } else {
      let biaseLayer = this.biases[Math.floor(Math.random()*this.biases.length)]
      biaseLayer = this.mutateArray(biaseLayer, mutationStrength)
    }
*/
  }

  mutateArray(arr, mutationStrength) {
    for (let i=0; i<arr.length; i++) arr[i] += (Math.random()-.5)*mutationStrength
    //arr[Math.floor(Math.random()*arr.length)] += (Math.random()-.5)*mutationStrength
    return arr
  }

  isArray(obj) {
    return typeof obj=="object"&&obj.toString()!=({}).toString()
  }

  clone() {
    let clone = new NeuralNetwork([1])
    clone.weights = structuredClone(this.weights)
    clone.biases = structuredClone(this.biases)
    clone.leakyReLUAlpha = this.leakyReLUAlpha
    clone.learnRate = this.learnRate
    clone.minError = this.minError
    clone.maxEpochs = this.maxEpochs
    return clone
  }
}
