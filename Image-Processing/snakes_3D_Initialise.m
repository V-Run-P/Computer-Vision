
function [almst,msh,bool,Options] = algo_snakes_3D(img)
%options for the snake algorithm
%img=one_cell;
Options=struct;
Options.Verbose=0;
Options.Iterations=1000;
Options.GIterations=500;
Options.Alpha=0.12;   %smoothness of contour_streching
Options.Beta=1;   %smoothness of contour_bending
Options.Kappa=0.3;   %weight for edges in the image
Options.Delta=0.04;   %balloon force
Options.Wline=0.04;
Options.Wterm=0;
Options.Mu=0.1;
Options.Gamma=1;
Options.Sigma1=0.7;
Options.Sigma2=1.0;
Options.Sigma3=1.0;

szey = size(img,1);
szex = size(img,2);
szez = size(img,3);
% thres = graythresh(img);
% Gbw = img>=round(255*thres);
se = strel('sphere',1);
Ie = imerode(img,se);
Iobr = imreconstruct(Ie,img);
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
%img=imreconstruct(img,strel('disk',2));
level = multithresh(Iobrcbr);%b=imbinarize(img,thres);
%shrp=imsharpen(b);
seg_I = imquantize(Iobrcbr,level);
figure
volshow(seg_I)

% Gbw=imdilate(seg_I,strel('disk',2));
% 
% Gfl =  reshape(imfill(reshape(Gbw,szey,szex*szez),'holes'),szey,szex,szez);
% Gfl = bwareaopen(Gfl,5,4);
% %Gfl
% bwsml = imdilate(imerode(Gfl,strel('disk',4)),strel('disk',4));

bwsml=seg_I;
%getting the boundary pixels
GX = circshift(bwsml,[1 0 0]);
GY = circshift(bwsml,[0 1 0]);
GZ = circshift(bwsml,[0 0 1]);

Xbound = abs(GX-bwsml); Ybound = abs(GY-bwsml); Zbound = abs(GZ-bwsml);
bnd = or(or(Xbound,Ybound),Zbound);
%bnd
%largest component
cmp = bwconncomp(bnd,18);
labeled = labelmatrix(cmp);
mx=1;
for nobj=2:cmp.NumObjects
    labeled(labeled == nobj) = mx;
end
S = regionprops(labeled ,'PixelIdxList');
bnd = false(szey,szex,szez); bnd(S.PixelIdxList)=true;
[bndx,bndy,bndz]=ind2sub([szey szex szez],find(bnd));

% 
% %starting surface: intial surface
conv = convhull(bndx,bndy,bndz);

bool=1;
if size(conv,1)>35000
bool=0;
end
    %whos
% [x,y,z]=sphere(26);
% sh=mesh(x*2+2,y*2+2,z*2+2);
% conv=sh;
%adjusting the image for removing noise while calculating gradients
adj = reshape(imadjust(reshape(img,szey,szex*szez)),[szey szex szez]);
thres = graythresh(adj);
%thres1 = adaptthresh(img);
%thres=imbinarize(img,thres);
almst = reshape(imadjust(reshape(adj,szey,szex*szez),[0.5*thres min(1.5*thres,1)]),[szey szex szez]);
almst = double(almst)/255;


msh.vertices = [bndx bndy bndz]; msh.faces=conv;
