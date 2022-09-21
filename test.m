ypred = predict(net,dsTest);


%Obtain pairs of noisy and pristine images from the test set using the preview function.
testBatch = preview(dsTest);

idx = 1;
y = ypred(:,:,:,idx);
x = testBatch{idx,1};
ref = testBatch{idx,2};
montage({x,y})