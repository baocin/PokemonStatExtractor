function feat = feature_extraction(img)
% Output should be a fixed length vector [1*dimension] for a single image. 
% Please do NOT change the interface.

    %Load model data once
    persistent trainedModel;
    if isempty(trainedModel)
       trainedModel = load('model.mat');
    end

    numberDimensions = 128;
    numNeighbors = 5;
    
    %TRY HSV and color
    grayscale = rgb2gray(img);
%     ycb = rgb2ycbcr(img);
%     grayscale = img(:,:,1);     %Take just the red channel
    %Interest point detection
        %Calculate Guassian at different scales
        %Store scale data
    %Feature Localization
        rawFeatures = detectSURFFeatures(grayscale, 'MetricThreshold', 1000, 'NumOctaves', 4, 'NumScaleLevels', 10);
    %Orientation - make rotation invariant
        %Calculate local gradients
            %Calculate Second Moments
            %m = moment(
    %Feature Descriptor
        %Store location, scale, and orientation of points
        [features, corners] = extractFeatures(grayscale, rawFeatures, ...
            'Method', 'SURF', ...
            'SURFSize', numberDimensions);       %TODO: test with 64 dimensions
        %[features, valid_points] = extractFeatures(I, points);
    %Keypoint Descriptor
        %16 gradient histograms, 8 bins each, weighted magnitude, smooth

        
     %Get the similarity (distance) between every feature and every
     %training 'word' center point
     distances = pdist2(double(features), double(trainedModel.centers), 'euclidean');
     
    %Map the features to their closest 'word' center points from the
    %training
    closestCenterIndex = [];
    for rowIndex = 1:size(distances,1)
       %Sort each row separately
       [sorted,centerIndicies] = sort(distances(rowIndex,:));
       
       %Save the closest 'word' center as the choice for this feature
        closestCenterIndex(rowIndex,1) = centerIndicies(1);
    end

     %Convert to a bag of words
     feat = histcounts(closestCenterIndex, size(trainedModel.centers,1));

end
