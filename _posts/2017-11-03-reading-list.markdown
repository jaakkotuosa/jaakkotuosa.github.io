---
layout: post
title:  "Reading list"
date:   2017-11-03 12:15:29 +0200
categories: papers
image: "https://jaakkotuosa.github.io/assets/images/screenshot.png"
---

Here's a list of papers and blog posts I have been reading, so that I may find them again someday. And maybe you can find them too.

## Generative Adversarial Networks
* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)
Isola et al 2016

   The paper for generating cat/bag/other images. Nice network.

## Water simulations
* [Data-Driven Fluid Simulations using Regression Forests](https://www.inf.ethz.ch/personal/ladickyl/fluid_sigasia15.pdf)
Ladický, Jeong et al 2015

   These regression forests look efficient, but it seems that they need to work quite a lot to find a formulation that was efficient. Far from throwing black box machine learning at the task.

* [Accelerating Eulerian Fluid Simulation With Convolutional Networks](https://arxiv.org/abs/1607.03597) Tompson et al 2017

   They found a neat formulation of the problem that gave a cost function that can used with unsupervised learning. Uses multiple resolutions to handle large scale effects.

## LSTM
* [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) Christopher Olah 2015

   Classic, I should read the other stuff too.
   It would be great if networks could learn to handle 
   the long term connections. The wikipedia syntax generation example
   shows that the network learned to close the parenthesis and format the code properly. 
   After trying things out with Keras, my undestanding
   is that as network is trained one batch of unrolled LSTM cells at time,
   the gradient (the learning) does not flow back to previous batch.
   Now does this mean that learning only happens within one batch and 
   longer connections are merely accidential generalizations? Need to find out more,
   and try out Keras stateful LSTM.
   

## Adversarial Examples 
* [Explaining And Harnnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf) 
Goodfellow et al (2015)

   Tries to explain why advesarial examples exist, and how to find them. Quite nice.

* [Synthesizing Robust Adversarial Examples](https://arxiv.org/abs/1707.07397)
Athalye et al (2017) 

   Creating 3D prints that consistenly fool Inception. Scary.

