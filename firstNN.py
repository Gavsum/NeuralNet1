import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import sys

# Data Cleansing object 
class DataCleanser(object):
    def __init__(self, train_features, train_targets, val_features, val_targets, test_features, test_targets, test_data, data):
        self.train_features = train_features
        self.train_targets = train_targets
        self.val_features = val_features
        self.val_targets = val_targets
        self.test_features = test_features
        self.test_targets = test_targets
        self.test_data = test_data
        self.data = data

    #Load data from csv returns as Pd.Frame
    def extract(self, data_path):
        rides = pd.read_csv(data_path)
        return rides

    def transform(self, rides):
        #Convert Categorical variables into dummy/indicator variabls in Pd.Frame
        dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
        for each in dummy_fields:
            dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
            rides = pd.concat([rides, dummies], axis=1)

        #purge non-required fields from data 
        fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                          'weekday', 'atemp', 'mnth', 'workingday', 'hr']
        self.data = rides.drop(fields_to_drop, axis=1)

    # Scale continuous variables in data such that mean = 0, std. deviation = 1
    # Also returns scaled features dict for use with plotting results
    def scale_feat(self):

        quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
        # Store scalings in a dictionary so we can convert back later
        scaled_features = {}
        for each in quant_features:
            mean, std = self.data[each].mean(), self.data[each].std()
            scaled_features[each] = [mean, std]
            self.data.loc[:, each] = (self.data[each] - mean)/std

        return scaled_features

    # Load up the data for use with training the model
    def load(self):
        # Save the last 21 days 
        data = self.data
        test_data = data[-21*24:]
        data = data[:-21*24]

        # Separate the data into features and targets
        target_fields = ['cnt', 'casual', 'registered']
        features, targets = data.drop(target_fields, axis=1), data[target_fields]
        test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

        # Hold out the last 60 days of the remaining data as a validation set
        train_features, train_targets = features[:-60*24], targets[:-60*24]
        val_features, val_targets = features[-60*24:], targets[-60*24:]

        return DataCleanser(train_features, train_targets, val_features, val_targets, test_features, test_targets, test_data, data)


class NeuralNetwork(object):    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1+np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        ### Forward pass ###
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # signals into final output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs = final_inputs
    
        ### Backward pass ###
        # Output layer error is the difference between desired target and actual output.
        output_errors = np.subtract(targets, final_outputs)
        
        # errors propagated to the hidden layer
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)

        # hidden layer gradients
        hidden_grad = hidden_outputs * (1 - hidden_outputs)
        
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)

        self.weights_input_to_hidden += self.lr * np.dot((hidden_grad * hidden_errors), inputs.T)
        
    def run(self, inputs_list):

        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  
        hidden_outputs = self.activation_function(hidden_inputs)

        #Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        final_outputs =  final_inputs
        
        return final_outputs


# Calc mean standard error
def MSE(y, Y):
    return np.mean((y-Y)**2)

# Set the hyperparameters here
def set_hyper(hy):
    #define hyper params and read into dict
    epochs = 1500 #start low and increase til error loss flattens out
    learning_rate = 0.01  # .005 to .02 works best 
    hidden_nodes = 25  # roughly half way between # of inputs and # of output units
    output_nodes = 1

    hy = {'epochs': epochs, 'learning_rate': learning_rate, 'hidden_nodes': hidden_nodes, 'output_nodes': output_nodes}

    return hy

# Display results of the model
def plot_display(losses, rides, network, scaled_features, dc):
    # Set up training loss vs validation loss graph
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)

    fig, ax = plt.subplots(figsize=(8,4))

    # Set up prediction vs actual values graph
    mean, std = scaled_features['cnt']
    predictions = network.run(dc.test_features)*std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((dc.test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    # Set date ranges for x axis
    dates = pd.to_datetime(rides.ix[dc.test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)

    # graph the graph! Woo!
    plt.show()


def run_train(dc, hy, network):
    losses = {'train':[], 'validation':[]}
    for e in range(hy['epochs']):
        # Go through a random batch of 128 records from the training data set
        #pdb.set_trace()
        batch = np.random.choice(dc.train_features.index, size=128)
        for record, target in zip(dc.train_features.ix[batch].values, 
                                  dc.train_targets.ix[batch]['cnt']):
            network.train(record, target)
        
        # Printing out the training progress
        train_loss = MSE(network.run(dc.train_features), dc.train_targets['cnt'].values)
        val_loss = MSE(network.run(dc.val_features), dc.val_targets['cnt'].values)
        
        sys.stdout.write("\rProgress: " + str(100 * e/float(hy['epochs']))[:4] \
             + "% ... Training loss: " + str(train_loss)[:5] \
             + " ... Validation loss: " + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)

    return losses


def main():
    # Initilize some stuffs
    data_path = 'Bike-Sharing-Dataset/hour.csv'
    # Hyper parameter dict
    hy = {}
    # Get hyper values
    hy = set_hyper(hy)
    # Dummy variable used to send a bunch of blank data frames to 
    # Data Cleansing function.
    # TODO: This seems stupid , pack all blank df into a dict and pass around
    df = pd.DataFrame()

    # Call data cleansing function, send objects to be filled with glorious data
    dc = DataCleanser(df,df,df,df,df,df,df,df)

    # Extract data from input into pd.Frame
    rides = dc.extract(data_path)

    # Perform necessary transforms on data for processing
    dc.transform(rides)

    # Scale / normalize features (0 mean, 1 std dev)
    scaled_features = dc.scale_feat()

    # Split data into relevant training / validation / predidiction sets
    dc = dc.load()

    # Define number of input nodes based on training_features
    N_i = dc.train_features.shape[1]

    # Triumphantly initilize our neural net!
    network = NeuralNetwork(N_i, hy['hidden_nodes'], hy['output_nodes'],\
    hy['learning_rate'])

    # Run it and output our loss as dict
    loss = run_train(dc, hy, network)
    
    # Render beautiful graphs of
    # Training loss vs validation loss and,
    # Predictions Vs. Outcomes
    plot_display(loss, rides, network, scaled_features, dc)
    

if __name__ == "__main__":
    main()