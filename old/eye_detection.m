% Please edit this function only, and submit this Matlab file in a zip file
% along with your PDF report

function [left_x, right_x, left_y, right_y] = eye_detection(img)

  %Get image dimensions
  [imageRows, imageColumns, numColorChannels] = size(img);

  % Bad Defaults
  left_x = (imageColumns/3);
  right_x = (imageColumns/3)*2;
  left_y = (imageRows/3);
  right_y = (imageRows/3);

  %=================================  RESIZE   ===============================
  % Resize to 1080p (respecting resolution)
  if (imageRows > 1024)
    img = imresize(img, 1024/imageRows);
  end

  %So we can convert back to the original image's coordinates later
  [newImageRows, newImageColumns, numColorChannels] = size(img);
  rowRatio = imageRows/newImageRows;
  columnRatio = imageColumns/newImageColumns;

  %=================================  YCbCr   ===============================
  ycbImage = rgb2ycbcr(img);
  blueDifferenceChannel = ycbImage(:,:,2);
  redDifferenceChannel = ycbImage(:,:,3);

  %Keep the thresholding consistent
  redDifferenceChannel = imcomplement(redDifferenceChannel);

  %Global Threshold
  otsuRedThreshold = graythresh(redDifferenceChannel);
  otsuBlueThreshold = graythresh(blueDifferenceChannel);

  %Even out the histogram so darks(The red regions) are more prominent
  redDifferenceChannel = histeq(redDifferenceChannel);
  blueDifferenceChannel = histeq(blueDifferenceChannel);

  %Adaptive Threshold
  adaptRedThreshold = adaptthresh(redDifferenceChannel,0.2);
  adaptBlueThreshold = adaptthresh(blueDifferenceChannel,0.5);

  redDifferenceChannel = medfilt2(redDifferenceChannel, [3 3]);
  redDifferenceChannel = imgaussfilt(redDifferenceChannel, 3);
  redDifferenceChannel = imdilate(redDifferenceChannel, strel('disk', 5));

  blueDifferenceChannel = medfilt2(blueDifferenceChannel, [3 3]);

  %Convert to a binary (black/white) image
  otsuRed = imbinarize(redDifferenceChannel,otsuRedThreshold);
  adaptRed = imbinarize(redDifferenceChannel,adaptRedThreshold);

  otsuBlue = imbinarize(blueDifferenceChannel,otsuBlueThreshold);
  adaptBlue = imbinarize(blueDifferenceChannel,adaptBlueThreshold);

  %merge the otsu and adaptive masks
  redDifferenceMask = adaptRed & otsuRed;
  blueDifferenceMask = adaptBlue & otsuBlue;

  %Not all eyes are completely red so expand the darks of the eyes a bit more
  redDifferenceMask = imclose(redDifferenceMask, strel('disk', 7));
  blueDifferenceMask = imclose(blueDifferenceMask, strel('disk', 13));

  %=================================  HSV   ===============================
  hsvImage = rgb2hsv(img);

  % Define thresholds
  %Delete any dark blues, deep purple, and other unlikely eye hues
  %http://infohost.nmt.edu/tcc/help/pubs/colortheory/web/hsv.html
  hueMax = .666;
  hueMin = .15;

  saturationMax = 1.000;
  saturationMin = 0.000;

  valueMax = 01.000;
  valueMin = 0.1000;

  valueChannel = hsvImage(:,:,3);
  valueChannel = imadjust(valueChannel);
  valueMax = graythresh(valueChannel);

  hsvMask = (hsvImage(:,:,1) >= hueMin ) & (hsvImage(:,:,1) <= hueMax) & ...
  (hsvImage(:,:,2) >= saturationMin ) & (hsvImage(:,:,2) <= saturationMax) & ...
  (hsvImage(:,:,3) >= valueMin ) & (hsvImage(:,:,3) <= valueMax);

  hsvMask = imdilate(hsvMask, strel('disk', 15));
  hsvMask = imclose(hsvMask, strel('disk',27));

  %=================================  COMBINE MASKS   ===============================
  %Merge all the separate masks together
  combinedMask = redDifferenceMask & blueDifferenceMask & hsvMask;% & labMask;

  %get the original image
  eyes = img;

  % Blank out the background(non-eye) regions with black
  eyes(repmat(~combinedMask,[1 1 3])) = 0;

  %=================================  EYE DETECTION   ===============================
  %Calculate the location of the eyes
  pImage = imdilate(eyes, strel('disk', 7));

  Rmin = 10;
  Rmax = 30;
  [centers, radii, metric] = imfindcircles(pImage,[Rmin Rmax], 'Sensitivity', 0.975, 'EdgeThreshold', 0.25);

  %filter out the circles to void ones that overlap or are too close to
  %eachother
  numCircles = size(centers);

  weightedCircles = centers(:, :);     %the x and y of the center of the circle
  weightedCircles(:, 3) = radii(:);
  weightedCircles(:, 4) = metric(:);

  %Sort by metric first (descending), radii size second (descending)
  weightedCircles = sortrows(weightedCircles, [-4]);

  %Limit to the top 100 circles - to keep computation down
  weightedCircles = weightedCircles(1:(min(100, end)), :);
  possibleEyeCircles = [];

  windowSize = 30;
  windowSizeHalf = windowSize * 0.5;

  for row = 1:size(weightedCircles, 1)
    circle = weightedCircles(row, :);
    circleX = circle(1,1);
    circleY = circle(1,2);

    %Crop to the circle's bounding box
    topLeftX = (circleX - windowSizeHalf);
    topLeftY = (circleY - windowSizeHalf);
    window = imcrop(pImage, [  topLeftX topLeftY windowSize windowSize ]);


    %Get hsv color of the window
    windowHsv = rgb2hsv(window);
    windowChannel = windowHsv(:, :, 1);

    %Get the mean hue of thic circle's boudning box
    meanValue = mean(windowChannel(:));

    %Count number of neighboring circles
    numCirclesInWindow = 0;
    for r = 1:size(weightedCircles, 1)
        %if y of this circle is in the window then count it
        testCircleY = weightedCircles(r, 2);
        if (testCircleY > circleY - windowSizeHalf && testCircleY < circleY + windowSizeHalf)
          numCirclesInWindow = numCirclesInWindow + 1;
        end
    end

    %Discount any circles that don't meet the criteria
    if (numCirclesInWindow > 2 && meanValue > 0.1)
      possibleEyeCircles(end+1, :) = circle;
      viscircles(circle(1, 1:2), circle(1, 3), 'Color','b');
  else
      viscircles(circle(1, 1:2), circle(1, 3), 'Color','r');
    end

    meanString = sprintf('%2.1f',numCirclesInWindow);
    text(circle(1,1)-35,circle(1,2)+13,meanString,'Color','y', 'FontSize',8,'FontWeight','bold');
  end

  %Check if number of circles is valid
  if (size(possibleEyeCircles, 1) >= 1)
      left_x = possibleEyeCircles(1, 1) * columnRatio;
      left_y = possibleEyeCircles(1, 2) * rowRatio;
  end
  if (size(possibleEyeCircles, 1) >= 2)
      right_x = possibleEyeCircles(2, 1) * columnRatio;
      right_y = possibleEyeCircles(2, 2) * rowRatio;
  end



end
