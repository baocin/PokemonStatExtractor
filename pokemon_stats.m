function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)

warning off;

%img, model
% img = imread('val/141_CP1215_HP81_SD2500_6026_35.png');
% model = false;

% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)

persistent imgNum;
if (isempty(imgNum))
    imgNum = 0;
end
imgNum = imgNum + 1;

assignin('base','model', model);

% % trainCPDigits = 
% load('trainCPDigits.mat');
% % trainSDDigits = 
% load('trainSDDigits.mat');
% % trainHPDigits = 
% load('trainHPDigits.mat');
% 
% assignin('base', 'trainCPDigits', trainCPDigits);
% assignin('base', 'trainSDDigits', trainSDDigits);
% assignin('base', 'trainHPDigits', trainHPDigits);
% %assignin('base', 'trainID', trainID);
% 
% load('observationLabels.mat');
% load('observations.mat');
% 
% assignin('base', 'observations', observations);
% assignin('base', 'observationLabels', observationLabels);
% 
% persistent observations;
% persistent observationLabels;
% persistent expected;
% persistent data;
% 
% if (isempty(observations))
%    observations = [];%zeros(10,59);
%    observationLabels = [];
%    expected = [];
%    data = table;
% end

% Replace these with your code


ID = model.defaultID;
CP = model.defaultCP;
HP = model.defaultHP;
stardust = model.defaultSD;
level = [327,165];
cir_center = [355,457];

%Resize the input image in order to avoid scale variance
standardSize = [ 1280 720 ];
sizeRatio = [ size(img,1)/standardSize(1) size(img,2)/standardSize(2) ];
img = imresize(img, standardSize);
peakResponseThreshold = 0.70;

%Cache the size since it'll be accessed a lot
inImageSize = size(img);

%Resize the mask to fit the current image
model.pokeMask = imresize(model.pokeMask, [ inImageSize(1) inImageSize(2) ]);
maskedRGBImage = bsxfun(@times, img, cast(model.pokeMask, 'like', img));
try
    maskedGrayImage = rgb2gray(maskedRGBImage);
catch E
   return; 
end

%Cut the masked Image into sections where specific templates will be found
oneFourthCol = round(inImageSize(2)/4);
oneTenthCol = round(inImageSize(2)/10);
one100thRow = round(inImageSize(1)/100);
oneThirdRow = round(inImageSize(1)/3);
oneHalfRow = round(inImageSize(1)/2);
oneSixthRow = round(inImageSize(1)/6);

cornerDetectionImage = maskedGrayImage(oneThirdRow:end-oneHalfRow, end-oneTenthCol:end);
textDetectionImage = maskedGrayImage(:,oneFourthCol:(inImageSize(2)-oneFourthCol));
pokeDetectionImage = maskedGrayImage((one100thRow * 15):(oneHalfRow-(one100thRow*5)), ...
        (oneTenthCol*2):(end-(oneTenthCol*2)));
    
%--------------------------  Detect CIR_Center ---------------------------
%Detect Corners (For center of the arc's X value)
corners = detectHarrisFeatures(cornerDetectionImage);
cornerPoint = corners.selectStrongest(1).Location;
cir_center(1) = round(inImageSize(2)/2) * sizeRatio(1);
cir_center(2) = round(cornerPoint(1) + oneThirdRow - (one100thRow*3)) * sizeRatio(2);

%--------------------------  Detect Dust ---------------------------
try
    % Crop to the very general area where the text appears
    bottomThird = textDetectionImage((oneThirdRow*2):end, :);
    
    %Template Match to get a point as a frame of reference
    [ dustLocation peakResponse ] = template_match(model.dustTemplate, bottomThird);
%     disp([ 'Peak Response:' num2str(peakResponse) ]);
    if (peakResponse > peakResponseThreshold )
        %Found the dust icon, now get the text region next to it
        dustStartRow = round(dustLocation(1));
        dustEndRow = round(dustLocation(1) + size(model.dustTemplate,1));
        dustStartCol = round(dustLocation(2) + size(model.dustTemplate, 2) * 1.2);
        dustEndCol = round(dustLocation(2) + size(model.dustTemplate,2) + (inImageSize(2) * 0.12));
        dustTextRegion = bottomThird(dustStartRow:dustEndRow, dustStartCol:dustEndCol);

        %Process the cropped text region for recognition
        dustTextRegion = imcomplement(dustTextRegion);
        dustTextRegion = imbinarize(dustTextRegion);

        %Find the rectangle bounding box of every digit character 
        croppedDust= cropCharacters(dustTextRegion);

        %Now detect the digits
        try
            sdStrings = [];
            for i = 1 : size(croppedDust, 2)
                
%                disp(sprintf('Loop #%d',i));
%                figure; imshow(croppedDust{i});
               img = imresize(croppedDust{i}, [ 21 14 ]); 
%                img = padarray(img, [ 5 5 ]);
               feat = extractHOGFeatures(img, 'CellSize', [2 2]);
               guessedDigit = predict(model.CPclassifier, feat);
%              guessedDigit = knnclassify(feat, cell2mat(trainSDDigits(:,2)), cell2mat(trainSDDigits(:,1)), 5);
               sdStrings = [ sdStrings num2str(guessedDigit)];
                
            end
            stardust = str2num(sdStrings);
        catch E
             disp('Used Default startdust');
             stardust = model.defaultSD;
         end
%     else
% %         disp('Invalid Template Match Position, typically happens if image is very small 175x288.');
% %        disp('Very weak template match - using default startdust value');
%        stardust = stardustDefault;
     end
catch E
%    disp('Very weak template match - using default startdust value');
   stardust = model.defaultSD;
end

disp([ 'StarDust:' num2str(stardust) ]);


%--------------------------  Detect HP ---------------------------
 try
	startRow = (cornerPoint(1) + oneThirdRow + (one100thRow * 10));
	endRow = (inImageSize(1)-(oneSixthRow*2) - (one100thRow * 10));
	middleThird = textDetectionImage(startRow:endRow, :);
	[ hpLocation peakResponse ] = template_match(model.slashWhiteTemplate, middleThird);
	
%     if (peakResponse > peakResponseThreshold)
		%Extract the text region
		hpStartRow = round(hpLocation(1))
        hpEndRow = min(floor(hpLocation(1) + size(model.slashWhiteTemplate,1) * 1.4), size(middleThird,1))
%             size(middleThird)
        hpTextRegion = middleThird(hpStartRow:hpEndRow, :);

        %Process the cropped text region for recognition
        hpTextRegion = imcomplement(imbinarize(hpTextRegion));
		[ croppedHP, bb ]= cropCharacters(hpTextRegion);
        
        %Now detect the digits
%         try
            HPStrings = [];
            
            numDigitsInHPValue = (size(croppedHP, 2) - 2 -1 ) / 2;
            
            centerColumn = size(hpTextRegion,2)/2;
            [ SLocation peakResponse ] = template_match(model.slashBinaryTemplate, hpTextRegion);
            
            disp(SLocation(1,2));
            
            slashIndex = 1;
            for bIndex = 1 : size(bb,1)
                disp(sprintf('Checking if slash coordinate is in bb #%d', bIndex));
                [ x y ] = bbCorners(bb(bIndex, 1:2), bb(bIndex,3:4));
                in = inpolygon(SLocation(2) + size(model.slashBinaryTemplate,2)/2, SLocation(1) + size(model.slashBinaryTemplate,1)/2, x, y);
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
            for i = hpStartDigit : hpEndDigit
               disp(sprintf('Loop #%d',i));
%                imwrite(croppedHP{i}, sprintf('hp/%d.png', i));
               
               img = imresize(croppedHP{i}, [ 21 14 ]); 
%                img = padarray(img, [ 5 5 ]);
%                figure; imshow(croppedHP{i});
%                figure; imshow(img);
               feat = extractHOGFeatures(img, 'CellSize', [2 2]);
           
               guessedDigit = predict(model.HPclassifier, feat);
%              guessedDigit = knnclassify(feat, cell2mat(trainSDDigits(:,2)), cell2mat(trainSDDigits(:,1)), 5);
               HPStrings = [ HPStrings num2str(guessedDigit)];
               
               
            end
            HP = str2num(HPStrings);
%         catch E
% % %             disp('Used Default HP');
%             HP =model.defaultHP;
%         end
    %else
%         disp('Very weak template match - using default HP value');
        %HP = HPDefault;
%     end
 catch E
% % %    disp('Very weak template match - using default HP value');
     HP = model.defaultHP;
 end

disp([ 'HP:' num2str(HP) ]);
% input('waiting for input');

%--------------------------  Detect CP ---------------------------
try
    cpGeneralRegion = textDetectionImage(1:oneSixthRow, :);
    % figure; imshow(cpGeneralRegion);
    [ cpLocation peakResponse ] = template_match(model.cpTemplate, cpGeneralRegion);

    if (peakResponse > peakResponseThreshold)
        cpStartRow = cpLocation(1) - (one100thRow * 2);
        cpEndRow = cpStartRow + (one100thRow * 6);
        cpStartCol = cpLocation(2) + size(model.cpTemplate,2) * 1.2;
        cpEndCol = cpStartCol + oneTenthCol*2.5;
        cpTextRegion = cpGeneralRegion(cpStartRow:cpEndRow, cpStartCol:cpEndCol);

        %Process the cropped text region for recognition
        cpTextRegion = imbinarize(cpTextRegion);
        croppedCP = cropCharacters(cpTextRegion);
        
        %Now detect the digits
        try
            cpStrings = [];
            for i = 1 : size(croppedCP, 2)
%                disp(sprintf('Loop #%d',i));
%                figure; imshow(croppedCP{i});
               img = imresize(croppedCP{i}, [ 21 14 ]); 
%                img = padarray(img, [ 5 5 ]);
               feat = extractHOGFeatures(img, 'CellSize', [2 2]);
               guessedDigit = predict(model.CPclassifier, feat);
%              guessedDigit = knnclassify(feat, cell2mat(trainSDDigits(:,2)), cell2mat(trainSDDigits(:,1)), 5);
               cpStrings = [ cpStrings num2str(guessedDigit)];
            end
            CP = str2num(cpStrings);
        catch E
%             disp('Used Default CP');
            CP =model.defaultCP;
        end
%     else
%         disp('Very weak template match - using default cp value');
%         CP = model.defaultCP;
    end
catch E
%    disp('Very weak template match - using default cp value');
    CP = model.defaultCP;
end

disp([ 'CP:' num2str(CP) ]);



fprintf('\n');

%--------------------------  Testing ---------------------------
% hFig = figure;
%   hAx  = axes;
%    figure;imshow(cpTextRegion);
%    hold on;
%    plot(cpLocation(1),cpLocation(2),'b*');
%    hold off;
%     pause(10);

% imwrite(maskedGrayImage, sprintf('mask/gray/maskedImage%d.png', i));
% imwrite(imbinarize(middleThird), sprintf('mask/edge/bw%d.png', i));
% imwrite(maskedRGBImage, sprintf('mask/RGB/maskedImage%d.png', i));

end


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