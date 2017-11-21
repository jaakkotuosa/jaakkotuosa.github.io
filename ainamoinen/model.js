'use strict'

KerasJS.Tensor.prototype.deepCopy = function () {
  return new KerasJS.Tensor(this.tensor.data, [this.tensor.shape[0]])
}

class Model {
  constructor () {
    this.temperature = 0.1
    this.predictionRunning = false
    this.abortPrediction = false
  }

  loadModel (baseUrl) {
    console.log('Loading model data...')
    console.time('load')
    this.model = new KerasJS.Model({
      filepaths: {
        model: baseUrl + 'inference_model.json',
        weights: baseUrl + 'inference_model_weights_weights.buf',
        metadata: baseUrl + 'inference_model_weights_metadata.json'
      },
      gpu: true
    })

    let indicesLoaded = $.getJSON(baseUrl + 'id2char.json').promise()
    indicesLoaded.then((indices) => {
      this.indexToChar = indices
      this.charToIndex = {}
      for (let i in this.indexToChar) {
        this.charToIndex[this.indexToChar[i]] = i
      }
      console.assert(Object.keys(this.indexToChar).length === Object.keys(this.charToIndex).length)
    })

    let allReady = Promise.all([this.model.ready(), indicesLoaded])
    allReady.then(
      () => {
        console.timeEnd('load')
        // check that we have 1-char inference this.model
        const shape = this.model.inputTensors.input.tensor.shape
        console.assert(shape.length === 1)
        console.assert(shape[0] === 1)
        this.inputData = {
          'input': new Float32Array(shape[0])
        }

        console.log('Ready to predict')
      },
      (err) => {
        console.error('Error', err)
      }
    )

    return allReady
  }

  clearModel () {
      // reset variables LSTM.js seems to use for its stateful operation. Not used by other layers
    for (let [k, v] of this.model.this.modelLayersMap) {
      delete v.currentHiddenState
      delete v.previousCandidate
    }
  }

  saveModelState () {
    let cache = {}
    for (let [k, v] of this.model.this.modelLayersMap) {
      if (v.currentHiddenState || v.previousCandidate) {
        cache[k] = {state: v.currentHiddenState.deepCopy(), candidate: v.previousCandidate.deepCopy()}
      }
    }
    return cache
  }

  loadModelState (cache) {
    for (let [k, v] of this.model.this.modelLayersMap) {
      if (cache[k]) {
        v.currentHiddenState.replaceTensorData(cache[k].state.tensor.data)
        v.previousCandidate.replaceTensorData(cache[k].candidate.tensor.data)
      }
    }
  }

  setTemperature (newTemperature) {
    this.temperature = newTemperature
  }

  _sample (probs) {
    for (let i = 0; i < probs.length; ++i) {
      probs[i] = Math.exp(Math.log(probs[i]) / this.temperature)
    }

    let multinomial = SJS.Multinomial(1, probs)
    let index = multinomial.draw().indexOf(1)
    return this.indexToChar[index]
  }

  * foo () {
    for (let i = 0; i < 3; i++) {
      yield i
    }
  }

  async _predictNext () {
    let outputData = await this.model.predict(this.inputData);
    let c = this._sample(outputData.output)

    this.inputData.input[0] = this.charToIndex[c]
    console.assert(this.indexToChar[this.inputData.input[0]] === c)

    console.log('_predictNextAsync', typeof c, c)
    return c
  }

  // TODO: generator can't be async. can't use await if not async
  *predict (inputText) {
    if (this.predictionRunning) {
      return
    }
    this.predictionRunning = true

    // clearModel();

    for (let offset = 0; offset < inputText.length; ++offset) {
      let c = inputText[offset]
      if (c in this.charToIndex) {
        this.inputData.input[0] = this.charToIndex[c]
      } else {
        console.log('unsupported char', c)
        this.inputData.input[0] = 0
      }

      // async function return value is automatically wrapped to Promise.resolve
      // so we need then to read it.
      // Also we cannot yield inside lambda function
      c = this._predictNext()
      yield c
    }

    // let cache = saveModelState();
    // console.log("Cache", JSON.stringify(cache));
    console.log('start prediction--------------------')

    for (let count = 0; count < 55; ++count) {
      let c;
      c = this._predictNext()
      yield c
    }

    this.predictionRunning = false
  }
}
