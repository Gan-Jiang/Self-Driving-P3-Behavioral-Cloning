This model is used for driving behavior cloning. Images and steering angles are recorded using simulator. These images form the training set. Then, train a convolutional neural network and predict the steering angles using it while self-driving in the simulator. The training dataset is obtained according to the following procedure: 

(1) I first tried my best to drive in the middle of the lane for three loops; 

(2) Then, I did the recovery trick for some difficult turns several times; 

(3) I put all of the center images I recorded to dataset. Then, a train-validation-test split was done, the ratio of the training data, validation data, and test data are 60%, 20% and 20% respectively;






The neural network architechture is obtained as follows:

(1) I set LeNet as the starting point. Since LeNet is using 32*32*3 pictures, I resize each image to 32*64*3. Then, I make all of the data zero-mean and normalized;

(2) I choose Adam as optimizer since it can consider the learning rate decay automatically; The batch size is set to 128 which is appropriate for my GPU memory. The number of epochs is set to 20 according to the results of ModelCheckpoint. The val_loss barely improve after 20 epochs. Also, this is a regression problem, so I set the loss function to MSE.

(3) Then, I start to try different architechture and parameters. I have tried changed the following:
	(a) Active function
	(b) The number of convolutional layers: 2, 3, 4
	(c) The depth and filter size of convolutional layers;
	(d) The number of fully-connected layers and neurons in the fully-connected layers. 
	(e) Drop-out
    I check the test accuracy, and also run the model using simulator. At last, I choose the model I use. 
Please provide a detailed description of the model architecture (number of layers, type of layers, orders of the layers) and the parameter settings for each layer ( dimensions of the layer, activation type). The description should be detailed enough to allow your readers to reconstruct a similar network. 
Optional, please consider to include a simple sketch that will demonstrate the general pattern with a detailed parameter settings of the architecture.





The model architecture is described as follows: 

(1)The number of layers are 11 layers;The order of the layers are convolutional layer, max-pooling layer, convolutional layer, max-pooling layer, convolutional layer, max-pooling layer, drop-out layer, and at last four fully-connected layers. 

(2) All activation functions are relu;

(3) The depth of three convolutional layers are 32, 64 and 96;

(4) The number of neurons for fully-connected layers are 120, 80, 43 and 1;

(5) After chose the model, I am using all the data to train the neural network again(including validation data and test data). The dimensions for each layer are as follows:
	conv1: input is 32*64*3, output is 29*61*32;
	pool1: 15*30*32
	conv2: 12*27*64
	pool2: 6*13*64
	conv3: 3*10*96
	pool3: 1*5*96
	fc1: 120
	fc2: 80
	fc3: 43
	fc4: 1

(6) The filter size for the convolutional layers are 4*4. The filter size for pooling layers is 2*2;

(7) The keep-prob is 0.5 for drop-out layer. 