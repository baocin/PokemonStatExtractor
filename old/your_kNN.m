function predict_label = your_kNN(feat)
%Given: The feature vector of an image to classify
% Output should be a fixed length vector [num of img, 1]. 
% Please do NOT change the interface.

%Load model data once
persistent trainedModel;
if isempty(trainedModel)
   trainedModel = load('model.mat');
end

%Number of nearest neighbors to consider when calculating repeated values
numNeighbors = 10;

%Get the similarity (distance) between every feature and every
%training 'word' center point
distances = pdist2(double(feat), double(trainedModel.bow), 'euclidean');

%Get the predicted label
nearest = [];
%Go through each testing image
for rowIndex = 1:size(distances,1)
   %Sort the training set by the testing image's bag of words
   [sorted,sortedIndicies] = sort(distances(rowIndex,:));
   sortedIndicies = sortedIndicies';        %Flip matrix
   %Use anonymous vectorized function to relabel the images of each row
   %into their corresponding category labels (retaining position in the
   %list)
   sortedIndicies = arrayfun(@(rowNum) trainedModel.labelTrain(sortedIndicies(rowNum)), 1:size(sortedIndicies,1))';
   
   %Take the highest occuring value
   predict_label(rowIndex, 1) = max(mode(sortedIndicies(1:numNeighbors)));
   nearest(rowIndex,:) = sortedIndicies;
end

end
