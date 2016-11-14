

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
CP = 123;
HP = 26;
stardust = 600;
level = [327,165];
cir_center = [355,457];

%Cache the size since it'll be accessed a lot
inImageSize = size(img)

%The cir_center will always be centered 
%Column(x position) will always be half of image width
cir_center(1) = round(inImageSize(2)/2);

%Mask over the entire card
pokeMask = im2bw(imread('mask.bmp'));
%Template of the dust logo
dustTemplate= rgb2gray(imread('dust.bmp'));
%Template of the edit pencil
editTemplate = rgb2gray(imread('edit.bmp'));

%Resize the mask to fit the current image
pokeMask = imresize(pokeMask, [ inImageSize(1) inImageSize(2) ]);
maskedRGBImage = bsxfun(@times, img, cast(pokeMask, 'like', img));
maskedGrayImage = rgb2gray(maskedRGBImage);

%Cut the masked Image into sections where specific templates will be found
oneThirdRow = round(inImageSize(1)/3);
middleThird = maskedGrayImage(oneThirdRow:(oneThirdRow*2), :);
bottomThird = maskedGrayImage((oneThirdRow*2):end, :);

% Find Dust 
dustLocation = template_match(dustTemplate, bottomThird);
dustLocation(2) = dustLocation(2) + oneThirdRow * 2
% pause(5);
    
% %Find Edit
% editLocation = template_match(editTemplate, middleThird);
% editLocation(2) = editLocation(2) + oneThirdRow
% % pause(5);

%Find Strongest horizontal line in the middle of the image
%(splits the 
    
imwrite(maskedGrayImage, sprintf('mask/gray/maskedImage%d.png', i));
imwrite(maskedRGBImage, sprintf('mask/RGB/maskedImage%d.png', i));

end


function [ position ] = template_match(template, background)
    correlation = normxcorr2(template, background);
%     figure, surf(correlation), shading flat
    [ypeak, xpeak] = find(correlation==max(correlation(:)));
    yoffSet = ypeak-size(template,1);
    xoffSet = xpeak-size(template,2);
    
    hFig = figure;
    hAx  = axes;
    imshow(background,'Parent', hAx);
    imrect(hAx, [xoffSet+1, yoffSet+1, size(template,2), size(template,1)]);
    
    position = [ xpeak(1) ypeak(1) ];
end