%%Load reduced stack with only cell information for active snakes

stack_n=3;

location_c={'cell-stack-locations'};

location_n={'cell-stack-locations'};

location = location_n{1, stack_n};
fileNames = dir(fullfile(location, '*.tif'));
for nf = 1:length(fileNames)
    filename = [fileNames(nf).folder '/' fileNames(nf).name];
    img = imread(filename);
    stk_n(:,:,nf) = img(:,:,2);
    disp(nf)
end

location = location_c{1, stack_n};
fileNames = dir(fullfile(location, '*.tif'));
for nf = 1:length(fileNames)
    filename = [fileNames(nf).folder '/' fileNames(nf).name];
    img = imread(filename); 
    stk_c(:,:,nf) = img(:,:,1);
    disp(nf)
end

for nf = 1:length(fileNames)
     stk_c(:,:,nf)= stk_c(:,:,nf) + 0.6.*stk_n(:,:,nf);
end


stats_2=regionprops('table',L,'all'); %L is the output of Watershed Algorithm
for cell_no=2:size(stats_2,1)
%cell_no=14;
figure;imshow(L==cell_no)
lo_x=floor(stats_2.BoundingBox(cell_no,1));
lo_y=floor(stats_2.BoundingBox(cell_no,2));
a=stats_2.BoundingBox(cell_no,3);
b=stats_2.BoundingBox(cell_no,4);

%Avoiding boundary condition
if lo_x == 0
    lo_x=1;
end
if lo_y ==0
    lo_y=1;
end


disp(cell_no);
one_cell = imcrop3(stk_c,[lo_x lo_y 1 a b nf-1]);
% figure; %volshow(one_cell)

[almst,msh,bool,Options]=algo_snakes_3D(one_cell);
%bool is an indicative of higher dimensional mesh and hence we downsample 
if bool==0
    d_one_cell=downsample(one_cell,2);
    [almst,msh,bool,Options]=algo_snakes_3D(one_cell);
    
end 

ov=Snake3D(almst,msh,Options);
% figure;h=patch(ov,'facecolor','r','edgecolor','k');

cells{1,cell_no-1}=ov;
cells{2,cell_no-1}=[lo_x,lo_y,a,b]; 
end