NEURAL NETWORK VISUALIZER

hey! this is an interactive web application that will help to demonstrates how a neural network learns to classify data through training , hidden layer transformations and decision boundaries.

This project was built to make neural network behaviour easier to understand using visual and interactive methods.
Users can generate synthetic datasets, configure a neural network, train the model and observe how the model separates different classes.

this will be the first version, I will do a better version called Version 2 !


## Project Overview

Neural networks can be difficult to understand because most of their learning process happens internally through weights, biases, and activations. This project helps make that process more visual.

The application allows users to:

- Select a synthetic dataset
- Configure the neural network architecture
- Train a PyTorch model
- View training loss and accuracy
- Visualize decision boundaries
- Inspect hidden-layer activations
- Review classification performance

The goal is to show how input data is transformed through hidden layers and how the model gradually learns to separate classes


## Features

- Interactive Streamlit web interface
- Dataset selection:
  - Moons
  - Circles
  - Blobs
- Adjustable neural network settings:
  - Number of hidden layers
  - Number of neurons per hidden layer
  - Activation function
  - Learning rate
  - Training epochs
- Neural network architecture visualization
- Training loss graph
- Accuracy progress graph
- Decision boundary visualization
- Hidden-layer activation heatmap
- Confusion matrix
- Classification report


## Tech Stack

- Python
- Streamlit
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

## Lets Try it out
1. Clone the Repo
git clone https://github.com/fiaabsinthe/Neural-Network-Visualizer.git
cd Neural-Network-Visualizer

2. Make sure you have all the tech stack installed
pip install -r requirements.txt

3. Once installed , run this in the terminal using:
streamlit run app.py

if streamlit isnt recognize or not added to PATH use this:
python -m streamlit run app.py

enjoy !
