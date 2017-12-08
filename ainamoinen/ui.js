let model = new Model()

function createDecoration () {
  $('.decoration').html('|/\\|<br>|\\/|<br>'.repeat(100))
}

function listenForChanges () {
  $('body').on('focus', '[contenteditable]', function () {
    var $this = $(this)
    $this.data('before', $this.html())
    return $this
  }).on('blur keyup paste input', '[contenteditable]', function () {
    var $this = $(this)
    if ($this.data('before') !== $this.html()) {
      $this.data('before', $this.html())
      $this.trigger('change')
    }
    return $this
  })

  $('#userInput').focus().on('change', function () {
    if ($(this).html().includes('<br>')) {
      let textToAdd = $(this).text() + $('#predicted').text() + '<br/>'
      $('#readyText').append(textToAdd)
      $(this).html('')
    } else {
      predict()
    }
  })

  $('#temperatureSlider').on('change', function () {
    model.setTemperature($(this).val())
    $('#userInput').focus()
    predict()
  })

  predict()
}

function predict () {
  $('#predicted').text('')

  let inputText = $('#readyText').text() + $('#userInput').text()
  inputText = inputText.toLowerCase()
  // replace &nbsp with normal space
  inputText = inputText.replace(/\u00A0/g, ' ')

  const prediction = model.predict(inputText)
  const step = (previousC) => {
    const res = prediction.next(previousC)
    if (res.done) {
      return
    }
    res.value.promise.then((c) => {
      if (res.value.isPredicted) {
        if (c === '\n') {
          if ($('#predicted').text()) {
            // avoid adding empty prediction
            $('#predicted').append('<br/>')
            step(null)
          }
        } else {
          if (c === ' ') {
            $('#predicted').append('&nbsp;')
          } else {
            $('#predicted').append(c)
          }
        }
      }
      $('#second').append(c)
      setTimeout(() => step(c), 0)
    })
  }

  step()
}

$(document).ready(function () {
  $('#userInput').focus()
  createDecoration()
  model.loadModel('model747.bin', 'char_indices.json').then(listenForChanges)
})
