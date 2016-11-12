

function [ID, CP, HP, stardust, level, cir_center] = pokemon_stats ()
%img, model
img = imread('val/141_CP1215_HP81_SD2500_6026_35.png');
% model = false;

% Please DO NOT change the interface
% INPUT: image; model(a struct that contains your classification model, detector, template, etc.)
% OUTPUT: ID(pokemon id, 1-201); level(the position(x,y) of the white dot in the semi circle); cir_center(the position(x,y) of the center of the semi circle)

persistent i;
if (isempty(i))
    i = 0;
end
i = i + 1;

persistent hmi;
if (isempty(hmi))
hmi = vision.MarkerInserter('Size', 10, ...
    'Fill', true, 'FillColor', 'Black', 'Opacity', 1);
end
persistent htm;
if (isempty(htm))
%     'SearchMethod', 'Three-step',
htm=vision.TemplateMatcher('Metric', 'Sum of squared differences', ...
'NeighborhoodSize', 15, 'BestMatchNeighborhoodOutputPort', true );
end

% Replace these with your code
ID = 1;
CP = 123;
HP = 26;
stardust = 600;
level = [327,165];
cir_center = [355,457];

maskImage = im2bw(imread('mask.bmp'));
dustImage = rgb2gray(imread('dust.bmp'));
editImage = imread('edit.bmp');

%Resize the mask to fit the current image
inImageSize = size(img)
% a = size(maskImage)
maskImage = imresize(maskImage, [ inImageSize(1) inImageSize(2) ]);


maskedRGBImage = bsxfun(@times, img, cast(maskImage, 'like', img));
maskedGrayImage = rgb2gray(maskedRGBImage);

% Find Dust 
% middleRow = round(inImageSize(1)/2);
bottomThirdImage = maskedGrayImage(round(inImageSize(1)/(3/2)):end, :);
Loc = step(htm, bottomThirdImage, dustImage)

% Mark the location on the image using white disc
  J = step(hmi, bottomThirdImage, Loc);

% imshow(bottomHalfImage); title('Template');
figure; imshow(J); title('Marked target');

% J = step(hmi, maskedImage, Loc);

% level = C;
% C = xcorr2(single(maskedRGBImage), single(dustImage))

imwrite(maskedGrayImage, sprintf('mask/gray/maskedImage%d.png', i));
imwrite(maskedRGBImage, sprintf('mask/RGB/maskedImage%d.png', i));

% imshow(maskedRGBImage); title('Template');
% figure; imshow(J); title('Marked target');

% exit()

end


