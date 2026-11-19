# athens-deeplearning

Course material and lab work for Practice in Deep Learning (TP40), an ATHENS week at Télécom Paris, November 2026. The `courses/` folder holds the lecture slides and the root holds the lab notebooks. All labs are written in PyTorch.

## Course chapters

- **Welcome**: course overview, teaching team and the schedule for the week.
- **Introduction to Python**: language basics needed for the labs.
- **Introduction to PyTorch**: tensors, autograd, datasets and dataloaders, defining models with `nn.Module`, training loops.
- **Introduction to Neural Networks**: machine learning approaches (supervised, unsupervised, reinforcement), the perceptron and the sigmoid neuron, weights and error, iterative training, linear vs non-linear classifiers, multilayer perceptrons.
- **Convolutional Neural Networks**: convolution filters, receptive field, stacking convolutional layers, max pooling, shift invariance and noise robustness, fully connected heads with softmax.
- **Deep Neural Networks**: going deeper than the basics, optimization issues (local minima, initialization), batch normalization, regularization, transfer learning and reusing pretrained feature extractors with new fully connected heads.
- **Recurrent Neural Networks**: sequence modeling, the different RNN configurations (many-to-one, many-to-many), LSTMs and word embeddings.

## Labs

### Lab 1: MNIST with a fully connected network
Loads MNIST with torchvision, normalizes it and builds dataloaders. A LeNet-300 style MLP (flatten, fully connected layers, LogSoftmax) is trained with NLLLoss for 20 epochs, with separate train and evaluation functions. Training and test loss and accuracy are tracked and plotted.

### Lab 2: CIFAR-10 with a CNN
The same pipeline moved to CIFAR-10, with data augmentation on the training set. The model is replaced with a convolutional network, and the training step adds L2 regularization on the weights. Results are shown as loss and accuracy curves plus a confusion matrix over the ten classes.

### Lab 3: Transfer learning
Classification of pizza, steak and sushi images (a small Food-101 subset, about 225 training images).
- A LeNet-5 adapted to RGB input is trained from scratch as a baseline and only gets slightly above chance.
- A pretrained EfficientNet-B0 is used as a frozen feature extractor with a new linear head (3,843 trainable parameters), reaching around 91% test accuracy.
- Extra assignments: confusion matrix and inspection of the most confident mistakes, testing the model on MNIST digits to see how it behaves on inputs outside its classes (it confidently calls them all sushi), and training the same EfficientNet from random weights to show that the pretrained weights, not the architecture, explain the result.

### Lab 4.1: Sentiment analysis with RNNs
Many-to-one prediction of movie review sentiment on the IMDB dataset, keeping the 5,000 most frequent words and padding or truncating reviews to 100 tokens.
- A simple model that learns word embeddings, averages them over time and feeds a sigmoid classifier.
- The learned embedding matrix is used to find the nearest words to a given word (for example "great").
- A second model replaces the averaging with an LSTM over the sequence and compares accuracy with the first.

## Data

`data/` contains the MNIST raw files and the pizza_steak_sushi image folders used by the labs. CIFAR-10 and IMDB are downloaded by the notebooks.
