function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)

warning off;

%img, model
% img = imread('val/141_CP1215_HP81_SD2500_6026_35.png');
% model = false;

% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)

persistent i;
if (isempty(i))
    i = 0;
end
i = i + 1;

load('observationLabels.mat');
load('observations.mat');

assignin('base', 'observations', observations);
assignin('base', 'observationLabels', observationLabels);
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
IDDefault = 1;
CPDefault = 10;
HPDefault = 40;
stardustDefault = 600;

ID = IDDefault;
CP = CPDefault;
HP = HPDefault;
stardust = stardustDefault;
level = [327,165];
cir_center = [355,457];

%Resize the input image in order to avoid scale variance
standardSize = [ 1280 720 ];
sizeRatio = [ size(img,1)/standardSize(1) size(img,2)/standardSize(2) ];
img = imresize(img, standardSize);
peakResponseThreshold = 0.70;

%Cache the size since it'll be accessed a lot
inImageSize = size(img);

%Mask over the entire card
pokeMask = im2bw(imread('mask.bmp'));
%Template of the dust logo
dustTemplate= rgb2gray(imread('dust.bmp'));
%Template of the edit pencil
editTemplate = rgb2gray(imread('edit.bmp'));
%Template of the / in the HP
slashTemplate = imread('slash.bmp');
%Template of the text "HP"
hpTemplate = imread('HP.bmp');
%Template of the text "P" in CP
cpTemplate = imread('CP.bmp');

%Resize the mask to fit the current image
pokeMask = imresize(pokeMask, [ inImageSize(1) inImageSize(2) ]);
maskedRGBImage = bsxfun(@times, img, cast(pokeMask, 'like', img));
% figure; imshow(maskedRGBImage);

maskedGrayImage = rgb2gray(maskedRGBImage);

%Cut the masked Image into sections where specific templates will be found
oneFourthCol = round(inImageSize(2)/4);
oneTenthCol = round(inImageSize(2)/10);
one100thRow = round(inImageSize(1)/100);
oneThirdRow = round(inImageSize(1)/3);
oneHalfRow = round(inImageSize(1)/2);
oneSixthRow = round(inImageSize(1)/6);

cornerDetectionImage = maskedGrayImage(oneThirdRow:end-oneHalfRow, end-oneTenthCol:end);
textDetectionImage = maskedGrayImage(:,oneFourthCol:(inImageSize(2)-oneFourthCol));

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
    [ dustLocation peakResponse ] = template_match(dustTemplate, bottomThird);
%     disp([ 'Peak Response:' num2str(peakResponse) ]);
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
            for i = 1 : size(croppedDust, 2)
                feat = extractLBPFeatures(croppedDust{i});
                guessedDigit = knnclassify(feat, observations, observationLabels);
                cpStrings = [ cpStrings num2str(guessedDigit)];
            end
            stardust = str2num(cpStrings);
        catch E
%             disp('Used Default startdust');
            stardust = stardustDefault;
        end
    else
%         disp('Invalid Template Match Position, typically happens if image is very small 175x288.');
%        disp('Very weak template match - using default startdust value');
       stardust = stardustDefault;
    end
catch E
%    disp('Very weak template match - using default startdust value');
   stardust = stardustDefault;
end

disp([ 'StarDust:' num2str(stardust) ]);


%--------------------------  Detect HP ---------------------------
try
	startRow = (cornerPoint(1) + oneThirdRow + (one100thRow * 10));
	endRow = (inImageSize(1)-(oneSixthRow*2) - (one100thRow * 10));
	middleThird = textDetectionImage(startRow:endRow, :);
	[ hpLocation peakResponse ] = template_match(slashTemplate, middleThird);
	
    if (peakResponse > peakResponseThreshold)
		%Extract the text region
		hpStartRow = round(hpLocation(1));
		hpEndRow = round(hpLocation(1) + size(slashTemplate,1) * 1.4);
		hpTextRegion = middleThird(hpStartRow:hpEndRow, :);

        %Process the cropped text region for recognition
        hpTextRegion = imcomplement(imbinarize(hpTextRegion));
		croppedHP = cropCharacters(hpTextRegion);
        
        %Now detect the digits
        try
            HPStrings = [];
            for i = 1 : size(croppedHP, 2)
                feat = extractLBPFeatures(croppedHP{i});
                guessedDigit = knnclassify(feat, observations, observationLabels);
                HPStrings = [ HPStrings num2str(guessedDigit)];
            end
            HP = str2num(HPStrings);
        catch E
%             disp('Used Default HP');
            HP =HPDefault;
        end
    else
%         disp('Very weak template match - using default HP value');
        HP = HPDefault;
    end
catch E
%    disp('Very weak template match - using default HP value');
    HP = HPDefault;
end

disp([ 'HP:' num2str(HP) ]);


%--------------------------  Detect CP ---------------------------
try
    cpGeneralRegion = textDetectionImage(1:oneSixthRow, :);
    % figure; imshow(cpGeneralRegion);
    [ cpLocation peakResponse ] = template_match(cpTemplate, cpGeneralRegion);

    if (peakResponse > peakResponseThreshold)
        cpStartRow = cpLocation(1) - (one100thRow * 2);
        cpEndRow = cpStartRow + (one100thRow * 6);
        cpStartCol = cpLocation(2) + size(cpTemplate,2) * 1.2;
        cpEndCol = cpStartCol + oneTenthCol*2.5;
        cpTextRegion = cpGeneralRegion(cpStartRow:cpEndRow, cpStartCol:cpEndCol);

        %Process the cropped text region for recognition
        cpTextRegion = imbinarize(cpTextRegion);
        croppedCP = cropCharacters(cpTextRegion);
        
        %Now detect the digits
        try
            cpStrings = [];
            for i = 1 : size(croppedCP, 2)
                feat = extractLBPFeatures(croppedCP{i});
                guessedDigit = knnclassify(feat, observations, observationLabels);
                cpStrings = [ cpStrings num2str(guessedDigit)];
            end
            CP = str2num(cpStrings);
        catch E
%             disp('Used Default CP');
            CP =CPDefault;
        end
    else
%         disp('Very weak template match - using default cp value');
        CP = CPDefault;
    end
catch E
%    disp('Very weak template match - using default cp value');
    CP = CPDefault;
end

disp([ 'CP:' num2str(CP) ]);



fprintf('\n');


% assignin('base', 'expected', expected);

% %Find Edit
% editLocation = template_match(editTemplate, middleThird);
% editLocation(2) = editLocation(2) + middleRow

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

function [ croppedCharacters ] = cropCharacters(croppedRegion)
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
