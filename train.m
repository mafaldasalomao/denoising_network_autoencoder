%noise salt and pepper

digitDatasetPath = fullfile(matlabroot,"toolbox","nnet", ...
    "nndemos","nndatasets","DigitDataset");
imds = imageDatastore(digitDatasetPath, ...
    IncludeSubfolders=true,LabelSource="foldernames");

%Specify a large read size to minimize the cost of file I/O.
imds.ReadSize = 500;
%Use the shuffle function to shuffle the digit data prior to training.
imds = shuffle(imds);
%Use the splitEachLabel function to divide imds into three image datastore
% containing pristine images for training, validation, and testing.
[imdsTrain,imdsVal,imdsTest] = splitEachLabel(imds,0.95,0.025);
%add noise
dsTrainNoisy = transform(imdsTrain,@addNoise);
dsValNoisy = transform(imdsVal,@addNoise);
dsTestNoisy = transform(imdsTest,@addNoise);

%combine image original and noised. expected by network

dsTrain = combine(dsTrainNoisy,imdsTrain);
dsVal = combine(dsValNoisy,imdsVal);
dsTest = combine(dsTestNoisy,imdsTest);

%normalize and resize
dsTrain = transform(dsTrain,@commonPreprocessing);
dsVal = transform(dsVal,@commonPreprocessing);
dsTest = transform(dsTest,@commonPreprocessing);

dsTrain = transform(dsTrain,@augmentImages);


%preview data
exampleData = preview(dsTrain);
inputs = exampleData(:,1);
responses = exampleData(:,2);
minibatch = cat(2,inputs,responses);
montage(minibatch',Size=[8 2])
title("Inputs (Left) and Responses (Right)")



%% Create network
imageLayer = imageInputLayer([32,32,1]);
%Create the encoding layers. Downsampling in the encoder is achieved by
% max pooling with a pool size of 2 and a stride of 2
encodingLayers = [ ...
    convolution2dLayer(3,8,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2), ...
    convolution2dLayer(3,16,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2), ...
    convolution2dLayer(3,32,Padding="same"), ...
    reluLayer, ...
    maxPooling2dLayer(2,Padding="same",Stride=2)];

%Create the decoding layers. The decoder upsamples the encoded signal using
% a transposed convolution layer with a stride of 2, which upsamples by a
% factor of 2. The network uses a clippedReluLayer as the final activation 
% layer to force outputs to be in the range [0, 1].
decodingLayers = [ ...
    transposedConv2dLayer(2,32,Stride=2), ...
    reluLayer, ...
    transposedConv2dLayer(2,16,Stride=2), ...
    reluLayer, ...
    transposedConv2dLayer(2,8,Stride=2), ...
    reluLayer, ...
    convolution2dLayer(1,1,Padding="same"), ...
    clippedReluLayer(1.0), ...
    regressionLayer];    

%Concatenate the image input layer, the encoding layers, and the decoding
% layers to form the convolutional autoencoder network architecture
layers = [imageLayer,encodingLayers,decodingLayers];

%options
options = trainingOptions("adam", ...
    MaxEpochs=50, ...
    MiniBatchSize=imds.ReadSize, ...
    ValidationData=dsVal, ...
    ValidationPatience=5, ...
    Plots="training-progress", ...
    OutputNetwork="best-validation-loss", ...
    Verbose=false);

%% Train the Network
net = trainNetwork(dsTrain,layers,options);
modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
save("trainedImageToImageRegressionNet-"+modelDateTime+".mat","net")
