# HARTS
Hardware Architecture Search (Harts), finds an optimal architecture of logic gates to perform a prediction.
The search space is discrete, but we make it continious by taking choices based on softmax over a trainable weights.
The weights can be trained using back propagation. A possible hardware architecture / arrangement of logic gates can be obtained by going greedy 
on the optimised weights.

DISCLAIMER : This is not a Binary Neural Network. This can be considered rather as an 'neural architecture search'.

# Classification of MNIST using logic gates
The above code can be used to generate optimal hardware architectures for prediction tasks. Here we present an 
example on the MNIST dataset. We were able to achieve a testing accuracy of 57% using just 5 layers of logic gates. The fact that such a search 
converges makes me happy :)

# TODO
  1. HDL generation from the trained networks
  2. Branch prediction for processor pipelining
