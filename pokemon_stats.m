

function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats (img, model)

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

% Replace these with your code
ID = 1;
CP = 10;
HP = 40;
stardust = 600;
level = [327,165];
cir_center = [355,457];

%Resize the input image in order to avoid scale variance
standardSize = [ 1280 720 ];
sizeRatio = [ size(img,1)/standardSize(1) size(img,2)/standardSize(2) ];
img = imresize(img, standardSize);

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
bottomThird = textDetectionImage((oneThirdRow*2):end, :);
dustLocation = template_match(dustTemplate, bottomThird);
%Found the dust icon, now get the text region next to it
dustStartRow = round(dustLocation(1));
dustEndRow = round(dustLocation(1) + size(dustTemplate,1));
dustStartCol = round(dustLocation(2) + size(dustTemplate, 2) * 1.2);
dustEndCol = round(dustLocation(2) + size(dustTemplate,2) + (inImageSize(2) * 0.12));
dustTextRegion = bottomThird(dustStartRow:dustEndRow, dustStartCol:dustEndCol);

%--------------------------  Detect HP ---------------------------
startRow = (cornerPoint(1) + oneThirdRow + (one100thRow * 10));
endRow = (inImageSize(1)-(oneSixthRow*2) - (one100thRow * 10));
middleThird = textDetectionImage(startRow:endRow, :);
hpLocation = template_match(slashTemplate, middleThird);
%Extract the text region
hpStartRow = round(hpLocation(1));
hpEndRow = round(hpLocation(1) + size(slashTemplate,1) * 1.4);
hpTextRegion = middleThird(hpStartRow:hpEndRow, :);

%--------------------------  Detect CP ---------------------------
cpGeneralRegion = textDetectionImage(1:oneSixthRow, :);
cpLocation = template_match(cpTemplate, cpGeneralRegion);
cpStartRow = cpLocation(1) - (one100thRow * 2);
cpEndRow = cpStartRow + (one100thRow * 6);
cpStartCol = cpLocation(2) + size(cpTemplate,2) * 1.2;
cpEndCol = cpStartCol + oneTenthCol*2.5;
cpTextRegion = cpGeneralRegion(cpStartRow:cpEndRow, cpStartCol:cpEndCol);

% %Find Edit
% editLocation = template_match(editTemplate, middleThird);
% editLocation(2) = editLocation(2) + middleRow

% detect_text(middleThird)
% pause(5);

% edgeImage = edge(middleThird, 'sobel', 0.05);

% slashLocation = template_match(slashTemplate, edgeImage);
% slashLocation(2) = slashLocation(2) + middleRow



%Find Strongest horizontal line in the middle of the image
%(splits the 

%--------------------------  Testing ---------------------------
%   hFig = figure;
%   hAx  = axes;
%     figure; imshow(cpGeneralRegion);
%    figure;imshow(cpTextRegion);
%   hold on;
%   plot(cpLocation(1),cpLocation(2),'b*');
%   hold off;
%    pause(10);

imwrite(maskedGrayImage, sprintf('mask/gray/maskedImage%d.png', i));
imwrite(middleThird, sprintf('mask/edge/edge%d.png', i));
imwrite(maskedRGBImage, sprintf('mask/RGB/maskedImage%d.png', i));

end


function [ position ] = template_match(template, background)
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
    position = [ yoffSet xoffSet ];
end
