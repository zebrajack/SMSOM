clear all
clc

source = '..\thermal\park\S3SOM2\';
dst = '..\thermal\park\S3SOMWithFilter2\';

filesName = dir(source);

for i = 3 : size(filesName, 1),
    tempImage = imread([source, filesName(i).name]);
    tempImageWithFilter = medfilt2(tempImage, [5, 5]);
    imwrite(tempImageWithFilter, [dst, filesName(i).name]);
end