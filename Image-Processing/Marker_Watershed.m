close all
clearvars
location_c={'cell-stack-locations'};

location_n={'nucleus-stack-locations'};

stack_n=4; %index to locations

%%For CELLS
location = location_c{1, stack_n}  ;
fileNames = dir(fullfile(location, '*.tif'));
for nf = 1:length(fileNames)
    filename = [fileNames(nf).folder '/' fileNames(nf).name];
    %C{k} = uint8(imread(filename));
    img = imread(filename);
    %img=im2gray(img);
    %uniqLevels = unique(img(:))
    %img = imsharpen(img,'Threshold',0.8);
    %img=imgradient(img);
    stk_c(:,:,nf) = img(:,:,1);
    
end

%%For Nucleus
location = location_c{1, stack_n};
fileNames = dir(fullfile(location, '*.tif'));
for nf = 1:length(fileNames)
    filename = [fileNames(nf).folder '/' fileNames(nf).name];
    %C{k} = uint8(imread(filename));
    img = imread(filename);
    %img=im2gray(img);
    %uniqLevels = unique(img(:))
    %img = imsharpen(img,'Threshold',0.8);
    stk_n(:,:,nf) = img(:,:,2);
    
end


% se_sphere = strel('disk',2);
% stk_n=imerode(stk_n,se_sphere);
% stk_n=imdilate(stk_n,se_sphere);
%stk_n=imnoise(stk_n,'gaussian');



% T=graythresh(stk_n);
% Ia=imbinarize(stk_n,T);
% Ia = uint8(Ia);
% I_n=Ia.*stk_n; %removed pixels values below threshold
%%^^^^^^^^^^^ This is done to get the pixel values only in particular
%%region.

for nf = 1:length(fileNames)
    stk(:,:,nf)= stk_c(:,:,nf) + stk_n(:,:,nf);
end

dm=zeros(size(img));
for i = 1:size(img,1)
    for j=1:size(img,2)
        dm(i,j)= mean(stk(i,j,:))/(std(double(stk(i,j,:))));
        
        %dm(i,j)= sum(stk(i,j,:));
    end
end
dm(isnan(dm)|isinf(dm))=0;
figure;
imshow(dm,[])
axis off
%title('Orignal Projection')
%export_fig(sprintf('Orignal_P%d',stack_n),'-tiff');
DM(:,:,stack_n)=dm(:,:,1);
dm_g=im2gray(dm);
figure;
imshow(dm_g,[])
axis off

%IG=imgradient(dm);
%figure;
%imshow(IG,[])
%title('Gradient Magnitude')


median_filter = medfilt2(dm_g,[2,2]);
I=median_filter;
figure;
imshow(I, []);
axis off
%title('Median Filtering')
% export_fig(sprintf('Median_F%d',stack_n),'-tiff');

conn=bwconncomp(I,18);
labeled = labelmatrix(conn);
stats = regionprops('table',conn,'all');
%mm=sqrt(1/numel(stats.EquivDiameter).*(sum(stats.EquivDiameter.^2)));
me=floor(mean(stats.Area));
med=floor(median(stats.Area));
mo=floor(mode(stats.Area));


se = strel('disk',7);
Ie = imerode(I,se);
Iobr = imreconstruct(Ie,I);
% figure;
% imshow(Iobr,[])
% title('Opening-by-Reconstruction')
% export_fig(sprintf('open_rec %d',stack_n),'-tiff');

% Imdilate followed by imreconstruct. Notice you must complement the image inputs and output of imreconstruct.
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure;
imshow(Iobrcbr,[])
axis off
% title('Opening-Closing by Reconstruction')
% export_fig(sprintf('Op_Cl_Re%d',stack_n),'-tiff');



fgm = imregionalmax(Iobrcbr);
% imshow(fgm)
% title('Regional Maxima of Opening-Closing by Reconstruction')

I2 = labeloverlay(I,fgm);
%imshow(fgm)
%title('Regional Maxima')
%figure;

conn=bwconncomp(fgm,18);
labeled = labelmatrix(conn);
stats_1 = regionprops('table',conn,'all');
disp(min(stats_1.Area));


%Notice that some of the mostly-occluded and shadowed objects are not marked,
%which means that these objects will not be segmented properly in the end result.
%Also, the foreground markers in some objects go right up to the objects' edge.
%That means you should clean the edges of the marker blobs and then shrink them a bit.
%You can do this by a closing followed by an erosion.
se2 = strel(ones(2,2));
fgm2 = imclose(fgm,se2);
fgm3 = imerode(fgm2,se2);
%imshow(fgm3)
%figure;
%This procedure tends to leave some stray isolated pixels that must be removed.
%You can do this using bwareaopen, which removes all blobs that have fewer than a certain number of pixels.
fgm4 = bwareaopen(fgm3,floor(0.3*min(stats_1.Area)));
I3 = labeloverlay(I,fgm4);
figure;
imshow(I3)
axis off
% title('Modified Regional Maxima')
% figure;

%thresh=graythresh(Iobrcbr);
%abw = adaptthresh(Iobrcbr);
bw = imbinarize(Iobrcbr);
figure;
imshow(bw)
% title('Thresholded Opening-Closing by Reconstruction')
%export_fig(sprintf('thres_op_cl_re %d',stack_n),'-tiff');
IG=imgradient(Iobrcbr);


D = bwdist(bw);%,'quasi-euclidean');


DL = watershed(D);

bgm = DL == 0;
figure;
imshow(bgm)
axis off
%title('Watershed Ridge Lines')
% figure;

gmag2 = imimposemin(IG, bgm | fgm4);

L = watershed(gmag2);
L_all(:,:,stack_n)=L(:,:,1);
labels = imdilate(L==0,ones(2,2)) + 2*bgm + 3*fgm4;

I4 = labeloverlay(dm,labels);

figure;
imshow(I4)
title('Markers and Object Boundaries')
export_fig(sprintf('Mark_Obj_Bnd %d',stack_n),'-tiff');


Lrgb = label2rgb(L,'jet','w','shuffle');
LRGB{1,stack_n}=Lrgb;
figure;
imshow(Lrgb)
axis off
title(sprintf('final-ws  %d',stack_n))
export_fig(sprintf('final_ws  %d',stack_n),'-tiff');
