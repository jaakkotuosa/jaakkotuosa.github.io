---
layout: post
image: "https://jaakkotuosa.github.io/assets/images/screenshot.jpg"
---

# AInamoinen - Kalevala text generation

See the demo [here](index.html).

## Background story

I was intriqued by [Andrej Karpathy's text generation samples](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 
because of the networks were able to learn long dependencies,
for example closing tags in wikipedia syntax.
So to get more familiar with LSTM I decided to try character by character text generation.
I wanted to see how well models could reproduce Kalevala metre poetry.

## Idea
I was envisoning nice interactive demo where people could seed the text generation with their own phrases,
and out comes Kalevala wisdom. After spending some time installing torch and other dependencies for char-rnn,
I started to appreciate that none of my contacts would be able to run a demo that 
would require Ubuntu and these requirements.
I needed an online demo. And browser-side demo would be simpler to deploy.

## Input data
I found material from [Harri Perälä's website](http://www.sci.fi/~alboin/trokeemankeli/kalevalamitta-aineistoja.htm).
Choosing [the stripped down version of Kalevala](http://www.iki.fi/harri.perala/trokeemankeli/kalevala_vain_sakeet.txt)
and [Kanteletar](http://www.iki.fi/harri.perala/trokeemankeli/kanteletar_karsittu.txt) 
I obtained [input material](./ainamoinen.txt) of around 1MB, not quite as large I was hoping for,
but I decided to go with it. To simplify the learning task, I lowercased the file and
removed some accidental characters (like '<'). In the end the vocabulary size was 37 characters. 

## Training
During the process I ran the training quite many times with different hyperparameters.
Two times I even got the model to diverge when using RMSprop optimizer, Adam seemed to work quite well out of the box 
(as advertized by [Andrew Ng](https://www.coursera.org/learn/deep-neural-network) ).

I was focusing quite a lot how well the models could correctly open and close quotes.
I was expecting something like:

> itse lausui, noin nimesi:
> "mi sinä olet miehiäsi,
> ku, kurja, urohiasi?
> vähän kuollutta parempi,
> katonutta kaunihimpi!"
> sanoi pikku mies merestä,

To my dismay even larger and deeper model failed to produce ':\n"' sequences,
or place the closing quotes before next quote.
 
I was also wondering why model insisted adding '--' or '...' every now and then,
and, right on, some of the input lines had these strings.
After this I checked the quotations in input, and yes, only some of them followed the pattern I was expecting.
The models were, it seems, doing quite nicely.

## Implementing the demo
Now, the browser-side JS neural networks do sound fancy, but the evaluation evaluation of the model shouldn't be that heavy,
and for my demo the speed would not be that crucial. It wouldn't be bad
if the user saw the computer to write the text character by character. 
What does internets offer?
- [Synaptic](https://github.com/cazala/synaptic) seems ambitious, browser based JS-only solution.
  I tried their LSTM example with slightly larger text and, well, it was very slow. 
- [TensorFire](https://tenso.rs/) is not out yet, but it seems to take the more feasible option of training offline
  (TensorFlow/torch) and only evaluating in browser with WebGL.
- Enter [Keras.js](https://github.com/transcranial/keras-js) - it evaluates [Keras](https://keras.io/) models in 
  browsers with GPU support. Yea!

[Keras LSTM example](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py) seemed to be a good
starting point.
It had a nice sampling function that allowed different "temperatures" used in Karpathy's samples.
I was thinking adjusting the creative freedom of the evaluation would be nice addition to the demo.  
This example however was stateless, (in Keras) meaning that cell states were reseted after each training and evaluation batch.
Being interested in long term capabilities (for example opening and closing long quotes) of LSTM 
I tried to turn it stateful. On the surface, in Keras this would seem to mean that one would need to use the same batch size
in training and in evaluation. 
For evaluation batch size 1 would be most suited, but for training that would be rather slow.

Luckily [yxtay](https://github.com/yxtay/char-rnn-text-generation) had realized how to work modify input shape for the evalution.
Unfortunately, keras-js could not load the small modified inference networks I exported from Keras.
After some debugging with local keras-js it turned out that weight names did have additional postfix "_1"
that keras-js did not expect. One probably could export the keras model somehow smartes,
but I resorted to removing offending postfix from json with sed.

### State management
State initialization was tricky. Keras does not export state, and what would be good state anyway?
With state initialized to zero the output was a mess for 20 or so first characters,
after which model started producing sensible output.

Interestingly, feeding valid seed text does not help. Only feeding the garbled output back in, the output becomes sensible.
This makes sense as this is how network has learned to bootstrap it's state during training.
For this stateful model initial wrong results are not that serious as it affects only the beginning, some 0.0001% of predictions.
This generates two hypotheses for later use:
Hypothesis 1: For stateless model, that has to init the state for every patch, the learning would probably drive the bootstrap happening faster.
Hypothesis 2: Some start-of-sequence marker could trigger more efficient init.

I ended up saving one assumed good state in json and initializing the model with that.
Then in the end of accepted line I save the model state and use that as initial state for next line.

## Failure
After the state management and number of bugs had been cleared I was eager to try it out.
Unfortunately the output looked very much like the model is not listening the used input.
The model being stateful means that during the training it practically always has the context from the previous words and sentences.
Also, during training the model never needs to adapt to novel seed texts.
So in short model is not trained to the interactive demo case. Quite classical mistake.

## Next
The stateful model can generate nice Kalevala text, so I think I'll keep it there, but next I'll revert back to stateless model
and see if that is more responsive to user input.
