---
layout: post
image: "https://jaakkotuosa.github.io/assets/images/screenshot.jpg"
---

# AInamoinen - Kalevala text generation

## Demo

## Background story

I was intriqued by [Andrej Karpathy's text generation samples](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 
because of the networks were able to learn long dependencies,
for example closing tags in wikipedia syntax.
So to get more familiar with LSTM I decided to try character by character text generation.

I wanted to see how well models could reproduce nelipolvi trokee.
I found material from [Harri Perälä's website](http://www.sci.fi/~alboin/trokeemankeli/kalevalamitta-aineistoja.htm).
Choosing [the stripped down version of Kalevala](http://www.iki.fi/harri.perala/trokeemankeli/kalevala_vain_sakeet.txt)
and [Kanteletar](http://www.iki.fi/harri.perala/trokeemankeli/kanteletar_karsittu.txt) 
I obtained [input material](./ainamoinen.txt) of around 1MB, not quite as large I was hoping for,
but I decided to go with it.

I was envisoning nice interactive demo where people could seed the text generation with their own phrases,
and out comes Kalevala wisdom. After spending some time installing torch and other dependencies for char-rnn,
I started to appreciate that none of my contacts would be able to run a demo that 
would require Ubuntu and these requirements.
I needed an online demo. And browser-side demo would be simpler to deploy.

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

While trying these different starting points I ran the training quite many times with different hyperparameters.
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
