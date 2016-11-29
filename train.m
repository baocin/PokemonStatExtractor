% You can change anything you want in this script.
% It is provided just for your convenience.
clear; clc; close all;

curDir = pwd
imgPath = fullfile(curDir, 'train');
imgListing = dir(imgPath);

numberOfClusters = 800;     %Word CLusters used in K-means
allFeatures = struct('features',{}, 'numFeatures',{}, 'fromImage',{}, 'fromLabel',{});

%Mask over the entire card
pokeMask = im2bw(imread('mask.bmp'));
%Template of the dust logo
dustTemplate= rgb2gray(imread('dust.bmp'));
%Template of the edit pencil
editTemplate = rgb2gray(imread('edit.bmp'));
%Template of the / in the HP
slashBinaryTemplate = imread('slashDivider.bmp');
slashWhiteTemplate = imread('slash.bmp');
%Template of the text "HP"
hpTemplate = imread('HP.bmp');
%Template of the text "P" in CP
cpTemplate = imread('CP.bmp');

HTemplate = imread('HTemplate.bmp');

defaultID = 133;
defaultCP = 12;
defaultHP = 100;
defaultSD = 200;

% Load all the features
% load('allFeatures.mat');
% load('justFeatures.mat');
% load('kmeans-200.mat');
% load('labelTrain.mat');
% load('bow.mat');
% load('bowLabel.mat');

stats = struct('ID', '', 'CP', '', 'HP', '', 'SD', '');
%observations = struct('ID', '', 'CP', '', 'HP', '', 'SD', '', 'Center', '');
trainHPDigits = {}; %column one is the true value, col 2 is feat vector
trainCPDigits = {};
trainSDDigits = {};
trainID = {};
standardSize = [ 1280 720 ];

load('trainCPDigits.mat');
load('trainSDDigits.mat');
load('trainHPDigits.mat');



for fileNumber = 4:size(imgListing,1)
    disp(sprintf('Processing Image Number %d', fileNumber));

    %Get Pokemon's Real Stats
    fileName = strcat(imgListing(fileNumber).name, '');
    splitName = strsplit(fileName, '_');

    %Convert to a number to remove leading 0s (since dont appear in imgs)
    stats(fileNumber).ID = str2num(splitName{1}); 
    stats(fileNumber).CP = str2num(splitName{2}(3:end));
    stats(fileNumber).HP = str2num(splitName{3}(3:end));
    stats(fileNumber).SD = str2num(splitName{4}(3:end));
    stats(fileNumber)

    %Load image info
    filePath = fullfile(imgPath, fileName);
    img = imread(filePath);

    %Resize the input image in order to avoid scale variance
    sizeRatio = [ size(img,1)/standardSize(1) size(img,2)/standardSize(2) ];
    img = imresize(img, standardSize);
    peakResponseThreshold = 0.70;

    %Cache the size since it'll be accessed a lot
    inImageSize = size(img);

    %Resize the mask to fit the current image
    pokeMask = imresize(pokeMask, [ inImageSize(1) inImageSize(2) ]);
    maskedRGBImage = bsxfun(@times, img, cast(pokeMask, 'like', img));
    maskedGrayImage = maskedRGBImage;
    try
        maskedGrayImage = rgb2gray(maskedRGBImage);
    catch E
       figure; imshow(maskedRGBImage);
       disp('Error opening Image!');
       %input('continue?');
       continue;
    end
    
    %figure; imshow(maskedGrayImage);

    %Cut the masked Image into sections where specific templates will be found
    oneFourthCol = round(inImageSize(2)/4);
    oneTenthCol = round(inImageSize(2)/10);
    one100thRow = round(inImageSize(1)/100);
    oneThirdRow = round(inImageSize(1)/3);
    oneHalfRow = round(inImageSize(1)/2);
    oneSixthRow = round(inImageSize(1)/6);

    cornerDetectionImage = maskedGrayImage(oneThirdRow:end-oneHalfRow, end-oneTenthCol:end);
    textDetectionImage = maskedGrayImage(:,oneFourthCol:(inImageSize(2)-oneFourthCol));
    pokeDetectionImage = maskedGrayImage((one100thRow * 15):(oneHalfRow-(one100thRow*5)),...
        (oneTenthCol*2):(end-(oneTenthCol*2)));
    %figure; imshow(cornerDetectionImage);
    %figure; imshow(textDetectionImage);
    %figure; imshow(pokeDetectionImage);
    
    %--------------------------  Detect CIR_Center ---------------------------
    
    disp('--------CIR Center-----------');
    %Detect Corners (For center of the arc's X value)
    corners = detectHarrisFeatures(cornerDetectionImage);
    cornerPoint = corners.selectStrongest(1).Location;
    cir_center(1) = round(inImageSize(2)/2) * sizeRatio(1);
    cir_center(2) = round(cornerPoint(1) + oneThirdRow - (one100thRow*3)) * sizeRatio(2);
    %disp(cir_center);
    
    
    %--------------------------  Detect Dust ---------------------------
    disp('--------Detecting Dust-----------');
    
    try
        % Crop to the very general area where the text appears
        bottomThird = textDetectionImage((oneThirdRow*2):end, :);

        %Template Match to get a point as a frame of reference
        [ dustLocation peakResponse ] = template_match(dustTemplate, bottomThird);
        %disp([ 'Peak Response:' num2str(peakResponse) ]);
        if (peakResponse > peakResponseThreshold )
            %Found the dust icon, now get the text region next to it
            dustStartRow = round(dustLocation(1));
            dustEndRow = round(dustLocation(1) + size(dustTemplate,1));
            dustStartCol = round(dustLocation(2) + size(dustTemplate, 2) * 1.2);
            dustEndCol = round(dustLocation(2) + size(dustTemplate,2) + (inImageSize(2) * 0.12));
            dustTextRegion = bottomThird(dustStartRow:dustEndRow, dustStartCol:dustEndCol);

            %Process the cropped text region for recognition
            dustTextRegion = imcomplement(dustTextRegion);
            dustTextRegion = imbinarize(dustTextRegion);

            %Find the rectangle bounding box of every digit character 
            croppedDust= cropCharacters(dustTextRegion);

            %Now detect the digits
            try
                cpStrings = [];
                
                %Check if digit segmentation matches up with the true value
                
                close all;
                trueDigits = num2str(stats(fileNumber).SD);
                if (size(croppedDust,2) == size(trueDigits, 2))
                   %Assume each digit matches with a single cropped digit
                   disp('digit number match')
                   for i = 1 : size(croppedDust, 2)
                       disp(sprintf('Loop #%d',i));
                       %figure; imshow(croppedDust{i});
                       trainSDDigits{fileNumber, 1} = trueDigits(i);
                       img = imresize(croppedDust{i}, [ 21 14 ]); 
%                        img = padarray(img, [ 5 5 ]);
                       feat = extractHOGFeatures(img, 'CellSize', [2 2]);
                       trainSDDigits{fileNumber, 2} = feat;
                       %extractLBPFeatures(croppedDust{i}); 
                   end
                end
%                  else
%                     disp('digit number mismatch')
%                     %ignore these samples
%                     
%                      for i = 1 : size(croppedDust, 2)
%                         disp(sprintf('Loop #%d',i));
%                         %figure; imshow(croppedDust{i});
%                         %digit = input('What digit is this?')
%                         %just for testing
%                         digit = '99999';
%                         trainSDDigits{fileNumber, 1} = digit;
%                         trainSDDigits{fileNumber, 2} = extractLBPFeatures(croppedDust{i});
%                        % input('Next?');
%                     end
%                  end
                
%                 %input('checking size')
%                 for i = 1 : size(croppedDust, 2)
% %                     feat = extractLBPFeatures(croppedDust{i});
%                     
% %                     guessedDigit = knnclassify(feat, observations, observationLabels);
%                     cpStrings = [ cpStrings num2str(guessedDigit)];
%                 end
%                 stardust = str2num(cpStrings);

                  stardust = 1;
            catch E
    %             disp('Used Default startdust');
                disp(E);
                stardust = defaultSD;
            end
        else
    %         disp('Invalid Template Match Position, typically happens if image is very small 175x288.');
    %        disp('Very weak template match - using default startdust value');
            disp(E);
           stardust = defaultSD;
        end
    catch E
    %    disp('Very weak template match - using default startdust value');
        disp(E);
       stardust = defaultSD;
    end

    disp([ 'StarDust:' num2str(stardust) ]);

    trainSDDigits(any(cellfun(@isempty,trainSDDigits),2),:) = [];
    
    
    
    %--------------------------  Detect HP ---------------------------
    disp('--------Detecting HP-----------');
    
    try
        startRow = round(cornerPoint(1) + oneThirdRow + (one100thRow * 10));
        endRow = round((inImageSize(1)-(oneSixthRow*2) - (one100thRow * 10)));
        middleThird = textDetectionImage(startRow:endRow, :);
        [ hpLocation peakResponse ] = template_match(slashWhiteTemplate, middleThird);

%         if (peakResponse > peakResponseThreshold)
            %Extract the text region
            hpStartRow = floor(hpLocation(1));
            hpEndRow = min(floor(hpLocation(1) + size(slashWhiteTemplate,1) * 1.4), size(middleThird,1));
%             size(middleThird)
            hpTextRegion = middleThird(hpStartRow:hpEndRow, :);

            %Process the cropped text region for recognition
            hpTextRegion = imcomplement(imbinarize(hpTextRegion));
            [ croppedHP bb ]= cropCharacters(hpTextRegion);

%             input(';');
            figure; imshow(hpTextRegion);
            input('');
            %Now detect the digits
%              try
                 HPStrings = [];
            
%             figure; imshow(hpTextRegion);
            numDigitsInHPValue = (size(croppedHP, 2) - 2 -1 ) / 2
            
            centerColumn = size(hpTextRegion,2)/2;
            [ SLocation peakResponse ] = template_match(slashBinaryTemplate, hpTextRegion);
            
            disp(SLocation(1,2));
            
            slashIndex = 1;
            for bIndex = 1 : size(bb,1)
                disp(sprintf('Checking if slash coordinate is in bb #%d', bIndex));
                [ x y ] = bbCorners(bb(bIndex, 1:2), bb(bIndex,3:4));
                in = inpolygon(SLocation(2) + size(slashBinaryTemplate,2)/2, SLocation(1) + size(slashBinaryTemplate,1)/2, x, y);
                if (in == 1)
                    slashIndex = bIndex
%                     figure; imshow(croppedHP{slashIndex});
%                     input('sdklfj');
                end
            end
            
            hpStartDigit = slashIndex + 1;
%             if (SLocation(1,2) < centerColumn)
%                %HP is to the right of the value
%                
%                 %HP to the left of the values
%                 hpStartDigit = 4
%             end
            
            hpEndDigit = hpStartDigit + numDigitsInHPValue - 1;
            
            
            trueDigits = num2str(stats(fileNumber).HP)
            
            %The cropped textregion has "28/28HP"
%             
%             "28/28HP"
%             "HP28/28"
%             "288/288HP"
%             "HP288/288"
            
%             figure; imshow();
            for i = hpStartDigit : hpEndDigit
               disp(sprintf('Loop #%d',i));
%                imwrite(croppedHP{i}, sprintf('hp/%d.png', i));
               
               img = imresize(croppedHP{i}, [ 21 14 ]); 
%                img = padarray(img, [ 5 5 ]);
%                figure; imshow(croppedHP{i});
%                figure; imshow(img);
               feat = extractHOGFeatures(img, 'CellSize', [2 2]);
               
%                disp(trueDigits(i));
               trainHPDigits{fileNumber, 1} = trueDigits(i-hpStartDigit+1);
               disp(trainHPDigits{fileNumber, 1})
               
               trainHPDigits{fileNumber, 2} = feat;
%                disp(trainHPDigits{fileNumber, 2})
               
               
%             input(';');
            end
%             input(';');
%                  HP = str2num(HPStrings);
               HP = 1;
%              catch E
%      %             disp('Used Default HP');
%                  disp(E);
%                  HP = defaultHP;
%              end
%          else
%              disp('Very weak template match - using default HP value');
             %disp(E);
%              HP = defaultHP;
%          end
     catch E
% %      %    disp('Very weak template match - using default HP value');
% %          disp(E);
         HP = defaultHP;
    end

    disp([ 'HP:' num2str(HP) ]);

    trainHPDigits(any(cellfun(@isempty,trainHPDigits),2),:) = [];
    
    
    
    %--------------------------  Detect CP ---------------------------
    disp('--------Detecting CP-----------');
    
    try
        cpGeneralRegion = textDetectionImage(1:oneSixthRow, :);
        % figure; imshow(cpGeneralRegion);
        [ cpLocation peakResponse ] = template_match(cpTemplate, cpGeneralRegion);

        if (peakResponse > peakResponseThreshold)
            cpStartRow = round(cpLocation(1) - (one100thRow * 2));
            cpEndRow = round(cpStartRow + (one100thRow * 6));
            cpStartCol = round(cpLocation(2) + size(cpTemplate,2) * 1.2);
            cpEndCol = round(cpStartCol + oneTenthCol*2.5);
            cpTextRegion = cpGeneralRegion(cpStartRow:cpEndRow, cpStartCol:cpEndCol);

            %Process the cropped text region for recognition
            cpTextRegion = imbinarize(cpTextRegion);
            croppedCP = cropCharacters(cpTextRegion);

            %Now detect the digits
            try
%                 cpStrings = [];

                close all;
                trueDigits = num2str(stats(fileNumber).CP);
                if (size(croppedCP,2) == size(trueDigits, 2))
                   %Assume each digit matches with a single cropped digit
                   disp('digit number match')
                   for i = 1 : size(croppedCP, 2)
                       disp(sprintf('Loop #%d',i));
                       %figure; imshow(croppedDust{i});
                       trainCPDigits{fileNumber, 1} = trueDigits(i);
                       img = imresize(croppedCP{i}, [ 21 14 ]); 
%                        img = padarray(img, [ 5 5 ]);
                       feat = extractHOGFeatures(img, 'CellSize', [2 2]);
                       trainCPDigits{fileNumber, 2} = feat;
%                        trainCPDigits{fileNumber, 2} = extractLBPFeatures(croppedCP{i}); 
                   end
                else
                    disp('digit number mismatch')
                end
                
%                 for i = 1 : size(croppedCP, 2)
%                     feat = extractLBPFeatures(croppedCP{i});
%                     guessedDigit = knnclassify(feat, observations, observationLabels);
%                     cpStrings = [ cpStrings num2str(guessedDigit)];
%                 end
%                 CP = str2num(cpStrings);

                CP = 1;
            catch E
    %             disp('Used Default CP');
                disp(E);
                CP =defaultCP;
            end
        else
    %         disp('Very weak template match - using default cp value');
            disp(E);
            CP = defaultCP;
        end
    catch E
    %    disp('Very weak template match - using default cp value');
        disp(E);
        CP = defaultCP;
    end

    disp([ 'CP:' num2str(CP) ]);
    trainCPDigits(any(cellfun(@isempty,trainCPDigits),2),:) = [];
    
    
    
    %--------------------- Train Pokemon ID ------------------------------
    %detectSURFeatures(pokeDetectionImage)
%     bagOfFeatures()
    trainID{fileNumber, 1} = stats(fileNumber).ID;
    grayscale = rgb2gray(pokeDetectionImage);
    rawFeatures = detectSURFFeatures(grayscale);
    [imgFeatures, corners] = extractFeatures(grayscale, rawFeatures, ...
    'Method', 'SURF', ...
    'SURFSize', 128);

    trainID{fileNumber, 2} = imgFeatures;
    
    
    
    fprintf('\n');

    
%    input('waiting for next image');
end

%}

%use the most frequent stat values as the default :)
% defaultID = mode(cell2mat({stats(:,2:end).ID}))
% defaultCP = mode(cell2mat({stats(:,2:end).CP}))
% defaultHP = mode(cell2mat({stats(:,2:end).HP}))
% defaultSD = mode(cell2mat({stats(:,2:end).SD}))

defaultID = 1;
defaultCP = 10;
defaultHP = 40;
defaultSD = 600;

HPclassifier = fitcecoc(cell2mat(trainHPDigits(:,2)), cell2mat(trainHPDigits(:,1)));
CPclassifier = fitcecoc(cell2mat(trainCPDigits(:,2)), cell2mat(trainCPDigits(:,1)));
SDclassifier = fitcecoc(cell2mat(trainSDDigits(:,2)), cell2mat(trainSDDigits(:,1)));

model = struct('defaultID', [], 'defaultCP', [], 'defaultHP', [], 'defaultSD', [], 'HPclassifier', [], 'CPclassifier', [], 'SDclassifier', [], 'pokeMask', [], 'dustTemplate', [], 'editTemplate', [], 'slashBinaryTemplate', [], 'hpTemplate', [], 'cpTemplate', [], 'HTemplate', [], 'slashWhiteTemplate', []);
model.CPclassifier = CPclassifier;
model.HPclassifier =HPclassifier;
model.SDclassifier = SDclassifier;
model.pokeMask = pokeMask;
model.dustTemplate = dustTemplate;
model.editTemplate = editTemplate;
model.slashBinaryTemplate = slashBinaryTemplate;
model.slashWhiteTemplate = slashWhiteTemplate;
model.hpTemplate = hpTemplate;
model.cpTemplate = cpTemplate;
model.defaultID = defaultID;
model.defaultCP = defaultCP;
model.defaultHP = defaultHP;
model.defaultSD = defaultSD;
model.HTemplate = HTemplate;

save('model.mat', 'model');

% 


% IDclassifier = fitcecoc(trainingFeatures, trainingLabels);
%CPclassifier = fitcecoc(trainingFeatures, trainingLabels);
%SDclassifier = fitcecoc(trainingFeatures, trainingLabels);










% 
% % SAVE ALL FEATURES OF EVERYIMAGE
%     %For each image
%     for j = 1:length(img_dir)
%         disp(['Processing image ', num2str(j), ' in folder ', num2str(l)]);
%         img = imread([imgPath,folderDir(l+2).name,'/',img_dir(j).name]);
%         current_row = (l-1)*imgPerClass+j;
%         
%   %%%%  imgFeatures = feature_extraction(img);
%           grayscale = rgb2gray(img);
%           rawFeatures = detectSURFFeatures(grayscale);
%           [imgFeatures, corners] = extractFeatures(grayscale, rawFeatures, ...
%                 'Method', 'SURF', ...
%                 'SURFSize', 128);       %TODO: test with 64 dimensions
% 
%         numFeatures = size(imgFeatures, 1);
%         
%         allFeatures(current_row).features = imgFeatures;
%         allFeatures(urrent_row).numFeatures = numFeatures;
%         allFeatures(current_row).fromImage = j;
%         allFeatures(current_row).fromLabel = l;
%     end
%     
% end
% % 
% save('allFeatures800.mat', 'allFeatures');
% save('labelTrain800.mat', 'labelTrain');
% 
% %Get just the features for every image
% % f = getfield(allFeatures, 'features', {1:end});
% allCells = {allFeatures(:).features}';
% justFeatures = [];
% for img=1:size(allFeatures(:), 1)
%    fprintf('Collecting image #%d features\n', img); 
% %    disp(allCells(img));
%     numFeatures = size(allCells{img}, 1);
%     sizeJustFeatures = size(justFeatures, 1);
%     ind = sizeJustFeatures+1;
%     ind2 = sizeJustFeatures+numFeatures;
%    justFeatures(ind:ind2, :) = allCells{img};
% end
% 
% save('justFeatures800.mat', 'justFeatures');
% 
% % K means on all features
% % Summarize image into visual words
% [clusterIndicies, centers, sum] = kmeans(justFeatures, numberOfClusters, 'MaxIter', 1000000);
% save('kmeans-800.mat', 'clusterIndicies', 'centers', 'sum');
% 
% 
% %Construct Bag of words for each image
% % Bag of Words
% bow = zeros(imgNum, numberOfClusters);      %preallocate cause we can
% featuresParsed = 0;
% 
% % %Loop through all images
% for imgID = 1: imgNum  %1 to 1800
%     numFeatures = allFeatures(imgID).numFeatures;
%     
%     featureMin = featuresParsed+1;
%     featureMax = numFeatures + featuresParsed;
%     
%     fprintf('feature Range: %d to %d\n', featureMin, featureMax);
%     
% %     a = justFeatures(featureMin:featureMax, :);
%     %Check which clusters each feature is from
%     featureToClusterMapping = clusterIndicies(featureMin:featureMax);
%     histogramFeatureClusterFreq = histcounts(featureToClusterMapping, numberOfClusters);
%     
%     %Save the counts into the histogram
%     bow(imgID, :) = histogramFeatureClusterFreq;
%     
%     %Normalize by dividing each by the number of features???
% %     bow(imgID) = histogramFeatureClusterFreq ./ numFeatures;
%     
%     featuresParsed = featureMax;
% end
% save('bow800.mat', 'bow');
% 
% %  %Train bow for each cluster
%  bowLabel = zeros(classNum, numberOfClusters);      %preallocate cause we can
%  for labelID = 1: classNum %1 to 30
%      clusterTotals = zeros(1, numberOfClusters);
%      for imgID = ((labelID-1)*imgPerClass+1) : (labelID * imgPerClass)
%         clusterTotals = clusterTotals(1,:) + bow(imgID,:);
%      end
%      %Save the counts into the histogram
%      bowLabel(labelID, :) = clusterTotals;
% end
% save('bowLabel800.mat', 'bowLabel');
% 
% %Post-Process on all images in database
%     %Inverted document frequency
% 
%     
% %Store results in .mat for later recall
% % assignin('base', 'allFeatures', allFeatures);
% 
% 
% %Save only what is needed 
% save('model800.mat', 'labelTrain', 'bow', 'centers', 'bowLabel');

disp('Done Training. Thank you for your patience');






function [ position, response] = template_match(template, background)
    correlation = normxcorr2(template, background);
%     figure, surf(correlation), shading flat
    [ypeak, xpeak] = find(correlation==max(correlation(:)));
    yoffSet = ypeak-size(template,1);
    xoffSet = xpeak-size(template,2);
    
%        hFig = figure;
%        hAx  = axes;
%        imshow(background,'Parent', hAx);
%         imrect(hAx, [xoffSet+1, yoffSet+1, size(template,2), size(template,1)]);
%     

    %TODO: Make confidence value by 
        % 1) calculating the average of the top 100 largest peaks
        % 2) getting the value of the maximum response peak
        % if less than 20% difference then consider it noise?
        
    response = correlation(ypeak,xpeak);
    position = [ yoffSet xoffSet ];
end

function [ croppedCharacters boundingBoxes ] = cropCharacters(croppedRegion)
%     closedRegion = imclose(croppedRegion, strel('square', 7));
    props = regionprops('table', croppedRegion, 'BoundingBox');
    boundingBoxes = round(table2array(props));
    
%     assignin('base', 'bb', boundingBoxes);
    
%     imshow(croppedRegion);
    croppedCharacters = cell(1, size(props, 1));
    for charID = 1 : size(props,1)
%         rectangle('Position', boundingBoxes(charID,:), 'edgecolor', 'red');
        croppedCharacters{charID} = croppedRegion( ...
            boundingBoxes(charID,2):boundingBoxes(charID,2)+boundingBoxes(charID,4)-1,  ...
            boundingBoxes(charID,1):boundingBoxes(charID,1)+boundingBoxes(charID,3)-1 ...
        );
    end
%     pause(5);
end

function [ xCoords yCoords ] = bbCorners(bbTopLeftCorner, bbWidth)
    %top left
    xCoords(1) = bbTopLeftCorner(1,1);
    yCoords(1) = bbTopLeftCorner(1,2);
    
    %top right
    xCoords(2) = bbTopLeftCorner(1,1) + bbWidth(1,2);
    yCoords(2) = bbTopLeftCorner(1,2);
    
    %bottom right
    xCoords(3) = bbTopLeftCorner(1,1) + bbWidth(1,2);
    yCoords(3) = bbTopLeftCorner(1,2) + bbWidth(1,1);
    
    %bottom left
    xCoords(4) = bbTopLeftCorner(1,1);
    yCoords(4) = bbTopLeftCorner(1,2) + bbWidth(1,1);
    
    
    
end