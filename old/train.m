% You can change anything you want in this script.
% It is provided just for your convenience.
clear; clc; close all;

imgPath = './train/';
classNum = 30;
imgPerClass = 60;
imgNum = classNum .* imgPerClass;
feat_dim = size(feature_extraction(imread('./val/Balloon/329060.JPG')),2);

folderDir = dir(imgPath);
labelTrain = zeros(imgNum,1);

numberOfClusters = 200;     %Word CLusters used in K-means

allFeatures = struct('features',{}, 'numFeatures',{}, 'fromImage',{}, 'fromLabel',{}); %cell(1,1);

% Load all the features
% load('allFeatures.mat');
% load('justFeatures.mat');
% load('kmeans-200.mat');
% load('labelTrain.mat');
% load('bow.mat');
% load('bowLabel.mat');

% SAVE ALL FEATURES OF EVERYIMAGE
% In each folder
for l = 1:length(folderDir)-2
    
    img_dir = dir([imgPath,folderDir(l+2).name,'/*.JPG']);
    if isempty(img_dir)
        img_dir = dir([imgPath,folderDir(l+2).name,'/*.BMP']);
    end
    
    labelTrain((l-1)*imgPerClass+1:l*imgPerClass) = l;

    %For each image
    for j = 1:length(img_dir)
        disp(['Processing image ', num2str(j), ' in folder ', num2str(l)]);
        img = imread([imgPath,folderDir(l+2).name,'/',img_dir(j).name]);
        current_row = (l-1)*imgPerClass+j;
        
  %%%%  imgFeatures = feature_extraction(img);
          grayscale = rgb2gray(img);
%           rawFeatures = detectSURFFeatures(grayscale, 'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 10);
          rawFeatures = detectSURFFeatures(grayscale);
          [imgFeatures, corners] = extractFeatures(grayscale, rawFeatures, ...
                'Method', 'SURF', ...
                'SURFSize', 128);       %TODO: test with 64 dimensions

        numFeatures = size(imgFeatures, 1);
        
        allFeatures(current_row).features = imgFeatures;
        allFeatures(current_row).numFeatures = numFeatures;
        allFeatures(current_row).fromImage = j;
        allFeatures(current_row).fromLabel = l;
    end
    
end
% 
save('allFeatures.mat', 'allFeatures');
save('labelTrain.mat', 'labelTrain');

%Get just the features for every image
% f = getfield(allFeatures, 'features', {1:end});
allCells = {allFeatures(:).features}';
justFeatures = [];
for img=1:size(allFeatures(:), 1)
   fprintf('Collecting image #%d features\n', img); 
%    disp(allCells(img));
    numFeatures = size(allCells{img}, 1);
    sizeJustFeatures = size(justFeatures, 1);
    ind = sizeJustFeatures+1;
    ind2 = sizeJustFeatures+numFeatures;
   justFeatures(ind:ind2, :) = allCells{img};
end

save('justFeatures.mat', 'justFeatures');

% K means on all features
% Summarize image into visual words
[clusterIndicies, centers, sum] = kmeans(justFeatures, numberOfClusters, 'MaxIter', 1000000);
save('kmeans-800.mat', 'clusterIndicies', 'centers', 'sum');


%Construct Bag of words for each image
% Bag of Words
bow = zeros(imgNum, numberOfClusters);      %preallocate cause we can
featuresParsed = 0;

% %Loop through all images
for imgID = 1: imgNum  %1 to 1800
    numFeatures = allFeatures(imgID).numFeatures;
    
    featureMin = featuresParsed+1;
    featureMax = numFeatures + featuresParsed;
    
    fprintf('feature Range: %d to %d\n', featureMin, featureMax);
    
%     a = justFeatures(featureMin:featureMax, :);
    %Check which clusters each feature is from
    featureToClusterMapping = clusterIndicies(featureMin:featureMax);
    histogramFeatureClusterFreq = histcounts(featureToClusterMapping, numberOfClusters);
    
    %Save the counts into the histogram
    bow(imgID, :) = histogramFeatureClusterFreq;
    
    %Normalize by dividing each by the number of features???
%     bow(imgID) = histogramFeatureClusterFreq ./ numFeatures;
    
    featuresParsed = featureMax;
end
save('bow.mat', 'bow');

%  %Train bow for each cluster
 bowLabel = zeros(classNum, numberOfClusters);      %preallocate cause we can
 for labelID = 1: classNum %1 to 30
     clusterTotals = zeros(1, numberOfClusters);
     for imgID = ((labelID-1)*imgPerClass+1) : (labelID * imgPerClass)
        clusterTotals = clusterTotals(1,:) + bow(imgID,:);
     end
     %Save the counts into the histogram
     bowLabel(labelID, :) = clusterTotals;
end
save('bowLabel.mat', 'bowLabel');

%Post-Process on all images in database
    %Inverted document frequency

    
%Store results in .mat for later recall
% assignin('base', 'allFeatures', allFeatures);


%Save only what is needed 
save('model.mat', 'labelTrain', 'bow', 'centers', 'bowLabel');

disp('Done Training. Thank you for your patience');