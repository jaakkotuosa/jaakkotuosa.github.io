'use strict'

KerasJS.Tensor.prototype.deepCopy = function () {
  return new KerasJS.Tensor(this.tensor.data, [this.tensor.shape[0]])
}

const blankMarker = String.fromCharCode(0)
const startMarker = String.fromCharCode(1)

class Model {
  constructor () {
    this.temperature = 0.1
    this.predictionRunning = false
  }

  loadModel (modelBin, charJson) {
    console.log('Loading model data...')
    console.time('load')
    this.model = new KerasJS.Model({
      filepath: modelBin,
      gpu: true
    })

    const indicesLoaded = $.getJSON(charJson).promise()
    indicesLoaded.then((indices) => {
      this.charToIndex = indices
      this.indexToChar = {}
      for (let i in this.charToIndex) {
        this.indexToChar[this.charToIndex[i]] = i
      }
      console.assert(Object.keys(this.indexToChar).length === Object.keys(this.charToIndex).length)
    })

    let allReady = Promise.all([this.model.ready(), indicesLoaded])
    allReady.then(
      () => {
        console.timeEnd('load')
        // check that we have 1-char inference this.model
        const shape = this.model.inputTensorsMap.get('input').tensor.shape
        console.assert(shape.length === 2)
        console.assert(shape[1] === Object.keys(this.charToIndex).length)
        this.inputShape = shape

        this.inputData = {
          'input': new Float32Array(shape.reduce((a, b) => a * b))
        }

        console.log('Ready to predict')
      },
      (err) => {
        console.error('Error', err)
      }
    )

    return allReady
  }

  setTemperature (newTemperature) {
    this.temperature = newTemperature
  }

  _sample (probs) {
    for (let i = 0; i < probs.length; ++i) {
      probs[i] = Math.exp(Math.log(probs[i]) / this.temperature)
    }
    probs[this.charToIndex[blankMarker]] = 0
    probs[this.charToIndex[startMarker]] = 0

    let multinomial = SJS.Multinomial(1, probs)
    let index = multinomial.draw().indexOf(1)
    return this.indexToChar[index]
  }

  // Returns promise for next predicted character
  async _predictNext () {
    let outputData = await this.model.predict(this.inputData)
    return this._sample(outputData.output)
  }

  // Returns a generator that returns promises for predicted characters
  * predict (inputText) {
    if (this.predictionRunning) {
      return
    }
    this.predictionRunning = true

    const seed = 'mieleni minun tekevi,\naivoni ajattelevi\nlähteäni laulamahan,\nsaa\'ani sanelemahan,\nsukuvirttä suoltamahan,\nlajivirttä laulamahan.\n'

    let filteredText = ''
    for (let c of inputText) {
      if (c in this.charToIndex) {
        filteredText += c
      } else {
        console.log(`ignoring unsupported char: ${c} (${c.charCodeAt(0)})`)
      }
    }
    inputText = seed + filteredText

    while (true) {
      // use only the tail of the input
      inputText = inputText.slice(-(this.inputShape[0] - 2))

      const inputTextAnnotated = (inputText + blankMarker).padStart(this.inputShape[0], startMarker)
      this.inputData.input.fill(0)
      for (let offset = 0; offset < inputTextAnnotated.length; ++offset) {
        const c = inputTextAnnotated[offset]
        if (c in this.charToIndex) {
          this.inputData.input[offset * this.inputShape[1] + this.charToIndex[c]] = 1
        }
      }
      const predictedChar = yield {isPredicted: true, promise: this._predictNext()}
      if (predictedChar === null) {
        break
      }
      inputText += predictedChar
    }
    this.predictionRunning = false
  }
}
