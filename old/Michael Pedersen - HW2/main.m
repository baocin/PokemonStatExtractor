%Michael Pedersen
%9/22/16


%Clear all other windows
clear; clc; close all;

%Low pass
iOne = imread('./source/plane.bmp');
originalOne = iOne;
% iOne = rgb2gray(iOne);

%High pass
iTwo = imread('./source/bird.bmp');
originalTwo = iTwo;
% iTwo = rgb2gray(iTwo);

%Save the original image dimensions
[row,col,depth] = size(iOne);
originalSize = [row col];

%How many layers of the pyramid
iterations = 7;

%Store each step in the process
lowSamples = cell(iterations);
highSamples = cell(iterations);

%Cutoffs
lowCutoff = 3;
highCutoff = 3;

%factor to blur each layer of the pyramid
blur = 1.5;

%Down to business - Gaussian pyramid
sample = originalOne;
for i=1:iterations
    lowSamples{i} = sample;
    blurred = imgaussfilt(sample, blur);

    %Include the rgb channels with ':'
    sample = blurred(1:2:end, 1:2:end, :);

    %Save the steps - debugging
    % f = figure('visible','off');
    % imshow(lowSamples{i}, 'Border', 'tight');
    % saveas(f,['low',int2str(i), '.jpg']);
end

%Laplacian Pyramid
sample = originalTwo;
for i=1:iterations
    blurred = imgaussfilt(sample, blur);
    highSamples{i} =  sample - blurred;

    %Include the rgb channels with ':'
    sample = sample(1:2:end, 1:2:end, :);

    %Save image for debugging
    % f = figure('visible','off');
    % imshow(highSamples{i}, 'Border', 'tight');
    % saveas(f,['high',int2str(i), '.jpg']);
end

%upsample all steps to the original image's size
for i=1:size(lowSamples, 2)
    lowSamples{i} =  imresize(lowSamples{i}, originalSize);
    highSamples{i} =  imresize(highSamples{i}, originalSize);

    %Save the steps - debugging
    f = figure('visible','off');
    imshow(lowSamples{i}, 'Border', 'tight');
    saveas(f,['low',int2str(i), '.jpg']);

    f = figure('visible','off');
    imshow(highSamples{i}, 'Border', 'tight');
    saveas(f,['high',int2str(i), '.jpg']);
end

%Merge the cutoff images from the high detail laplacian pyramid
final = highSamples{highCutoff};
for hc=1:highCutoff-1
    final = uint8(final) + uint8(highSamples{hc} .* (1/highCutoff));
end

%Merge low detail gaussian pyramid images
for lc=lowCutoff:iterations
    disp(lc)
    final = uint8(final) + uint8(lowSamples{lc} .* (1/(iterations - lowCutoff+1)));
end

%Save the final image
f = figure('visible','off');
imshow(final, 'Border', 'tight');
saveas(f,['final.jpg']);


%Old way of merging
% final = highSamples{3} .* (1/numHighSamples) + highSamples{2} .* (1/numHighSamples) + ...
% lowSamples{2} .* (1/numLowSamples) + lowSamples{3} .* (1/numLowSamples) + lowSamples{4} .* (1/numLowSamples) + lowSamples{5} .* (1/numLowSamples) + lowSamples{6} .* (1/numLowSamples) + lowSamples{7} .* (1/numLowSamples) + lowSamples{8} .* (1/numLowSamples);
