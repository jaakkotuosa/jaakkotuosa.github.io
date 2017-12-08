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
    // dev time flag to generate good initial state for model
    this.createInitialState = false
    this.isInInitialCondition = true
  }

  // stateful=true here means that is was trained with stateful keras_model.py
  // stateful=false means lstm_text_generation.py
  loadModel (stateful, baseUrl) {
    console.log('Loading model data...')
    console.time('load')
    this.stateful = stateful
        /*
    this.model = new KerasJS.Model({
      filepaths: {
        model: baseUrl + 'model110.json',
        weights: baseUrl + 'model110_weights.buf',
        metadata: baseUrl + 'model110_metadata.json'
      },
      gpu: true
    })*/
    this.model = new KerasJS.Model({
      filepath: './model17.bin',
      gpu: true
    })


    const indicesLoaded = $.getJSON(baseUrl + (this.stateful ? 'id2char.json' : 'char_indices.json')).promise()
    indicesLoaded.then((indices) => {
      if (this.stateful) {
        this.indexToChar = indices
        this.charToIndex = {}
        for (let i in this.indexToChar) {
          this.charToIndex[this.indexToChar[i]] = i
        }
      } else {
        this.charToIndex = indices
        this.indexToChar = {}
        for (let i in this.charToIndex) {
          this.indexToChar[this.charToIndex[i]] = i
        }
      }
      console.assert(Object.keys(this.indexToChar).length === Object.keys(this.charToIndex).length)
    })

    let promises = [this.model.ready(), indicesLoaded]
    if (!this.createInitialState && this.stateful) {
      const initialStateLoaded = $.getJSON(baseUrl + 'initial_state.json').promise()
      promises.push(initialStateLoaded)

      initialStateLoaded.then((state) => {
        function tensorFromObject (obj) {
          if (obj) {
            return new KerasJS.Tensor(Object.values(obj.tensor.data), obj.tensor.shape)
          }
          return undefined
        }

        // serialized tensor data Float32Array seems to be an object {0: first value, ...}
        // replace objects with proper tensors
        for (let k in state) {
          state[k].state = tensorFromObject(state[k].state)
          state[k].candidate = tensorFromObject(state[k].candidate)
        }
        this.stateInTheBeginningOfLine = state
      })
    }

    let allReady = Promise.all(promises)
    allReady.then(
      () => {
        console.timeEnd('load')
        // check that we have 1-char inference this.model
        const shape = this.model.inputTensorsMap.get("input").tensor.shape
        if (this.stateful) {
          console.assert(shape.length === 1)
        } else {
          console.assert(shape.length === 2)
          console.assert(shape[1] === Object.keys(this.charToIndex).length)
        }
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

  _clearModel () {
       // console.assert(false)
      // reset variables LSTM.js seems to use for its stateful operation. Not used by other layers
    for (let [k, v] of this.model.modelLayersMap) {
      delete v.currentHiddenState
      delete v.previousCandidate
    }
  }

  _saveModelState () {
        console.assert(false)
    let state = {}
    for (let [k, v] of this.model.modelLayersMap) {
      if (v.currentHiddenState || v.previousCandidate) {
        state[k] = {state: v.currentHiddenState.deepCopy(), candidate: v.previousCandidate.deepCopy()}
      }
    }
    return state
  }

  startPredictionsFromCurrentState () {
    this.stateInTheBeginningOfLine = this._saveModelState()
  }

  _loadModelState (state) {
    console.assert(false)
    for (let [k, v] of this.model.modelLayersMap) {
      if (state[k]) {
        if (v.currentHiddenState) {
          // let model take care of updating webgl texture
          v.currentHiddenState.replaceTensorData(state[k].state.tensor.data)
        } else {
          // unitialized model
          v.currentHiddenState = state[k].state.deepCopy()
        }

        if (v.previousCandidate) {
          v.previousCandidate.replaceTensorData(state[k].candidate.tensor.data)
        } else {
          v.previousCandidate = state[k].candidate.deepCopy()
        }
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
    probs[this.charToIndex[blankMarker]] = 0
    probs[this.charToIndex[startMarker]] = 0

    let multinomial = SJS.Multinomial(1, probs)
    let index = multinomial.draw().indexOf(1)
        console.log("got", index, this.indexToChar[index])
    return this.indexToChar[index]
  }

  _setInput(c) {
    const index = this.charToIndex[c]
        if (c == startMarker) {
            console.log("setinput <start>", index)
        } else if (c == blankMarker) {
            console.log("setinput <blank>", index)
        } else {
            console.log("setinput", index, c)
        }
    if (this.stateful) {
     // this.inputData.input[0] = index
    } else {
      this.inputData.input.fill(0)
    //  this.inputData.input[index] = 1.0/index
   //   this.inputData = {
   //     'input': new Float32Array(36)
    //  }
            this.inputData.input[index] = 1
    }
  }

  // Returns promise for next predicted character
  async _predictNext () {
       // console.log("x", this.inputData.input[34], this.inputData.input[35])
    let outputData = await this.model.predict(this.inputData)
    this.isInInitialCondition = false
    //console.log("y", outputData.output[0], outputData.output[1], outputData.output[2], outputData.output[3], outputData)
    return this._sample(outputData.output)
  }

  // Returns a generator that returns promises for predicted characters
  * predict (inputText) {
    if (this.predictionRunning) {
      return
    }
    this.predictionRunning = true

    const seed = 'mieleni minun tekevi,\naivoni ajattelevi\nlähteäni laulamahan,\nsaa\'ani sanelemahan,\nsukuvirttä suoltamahan,\nlajivirttä laulamahan.\n'
    if (this.createInitialState) {
      // use the Kalevala opening words as seed to the state
      inputText = seed
    } else {
      if (this.stateful) {
        this._loadModelState(this.stateInTheBeginningOfLine)
      } else {
        // TODO: do we need this?
        // this._clearModel()
      }

      if (!inputText) { // && this.isInInitialCondition) {
        // if there is not input text and model is it's initial state,
        // its probably better to feed in some something familiar instead of null character.
        inputText = seed
      }
    }


    console.assert(this.charToIndex[startMarker] == 35)
        console.assert(this.charToIndex[blankMarker] == 34)

    if (true) {
//        inputText = "väi"
        while (true) {
            inputText = inputText.slice(-(this.inputShape[0] - 2))

            console.log("input", inputText)
            const inputTextAnnotated = startMarker.repeat(this.inputShape[0] - 1 - inputText.length) + inputText + blankMarker // inputText
            this.inputData.input.fill(0)
            for (let offset = 0; offset < inputTextAnnotated.length; ++offset) {
              const c = inputTextAnnotated[offset]
              if (c in this.charToIndex) {
                const index = this.charToIndex[c]
                        /*
                        if (c == startMarker) {
                            console.log("setinput <start>")
                        } else if (c == blankMarker) {
                            console.log("setinput <blank>")
                        } else {
                            console.log("setinput", index, c)
                        }
                        */
                this.inputData.input[offset * this.inputShape[1] + index] = 1
              } else {
                console.log('unsupported char', c, c.charCodeAt(0))
                continue
              }
            }
                /*
            console.log(this.inputData.input)
                console.log(this.refX)
            console.assert(this.inputData.input.length === this.refX.length)
            for (let i = 0; i < this.inputData.input.length; ++i) {
                    if (this.inputData.input[i] !== this.refX[i]) {
                        console.log(i, this.inputData.input[i], this.refX[i])
                        console.assert(false)
                        break
                    }

            }
    */
            const predictedC = yield {isPredicted: true, promise: this._predictNext()}
                if (predictedC === null) {
                  break
                }
            inputText += predictedC
        }
    } else {
            inputText = "v"
        inputText = startMarker.repeat(this.inputShape[0] - 1 - inputText.length) + inputText + blankMarker // inputText

        for (let offset = 0; offset < inputText.length; ++offset) {
          if (!this.createInitialState) {
            let c = inputText[offset]

            if (c in this.charToIndex) {
              this._setInput(c)
            } else {
              console.log('unsupported char', c, c.charCodeAt(0))
              continue
            }
          }
          // yield also these seed results so that promise can be waited
          // before next char is fed in
          yield {isPredicted: !(offset + 1 < inputText.length), promise: this._predictNext()}
        }

        let prevC
        for (let count = 0; ; ++count) {
          if (count > 34) {
            // make sure prediction completes at some point
            // and make sure it ends with new line
            yield {isPredicted: true, promise: Promise.resolve('\n')}
            break
          }

          this._setInput(blankMarker)
          const c = yield {isPredicted: true, promise: this._predictNext()}
          if (this.createInitialState) {
            // wait until model has stabilized (hopefully after 100 chars)
            // and then wait for a phrase end,
            // which is hopefully good state to enter some user input
            if ((prevC === '.') && (c === '\n') && (count > 100)) {
              let state = this._saveModelState()
              console.log('Initial state:', JSON.stringify(state))
              break
            }
          } else {
            // null is marker to stop predicting
            if (c === null) {
              break
            }
          }
          prevC = c
        }
    }
//*/
    this.predictionRunning = false
  }
}
