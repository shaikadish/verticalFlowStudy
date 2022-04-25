% Main algorithm function

% I -> input: Input image
% end_value -> input: tunable parameter for polygonization
% blur -> input: tunable parameter for image blur
% conc_sens -> input: tunable parameter for algorithms sensitivity to noisy
% concavities
% save_path -> input: path to where properties are saved
% props -> output: structured array with properties of detected ellipses

function [props]=bp_algorithm(I,end_val,blur,conc_sens,save_path)
%% Pre-processing

imshow(I,'InitialMagnification','fit')
hold on

Iblur=imgaussfilt(I,blur); %Gaussian blur to remove noise and minor concavities
imbin=~imbinarize(Iblur,0.5); % Background foreground segmentation using binarization
imbin=imfill(imbin,'holes'); % Filling of holes in binary image


%% Distance based Watershed transform

D=bwdist(~imbin); % Distance transform
D=-D; % Complement of distance transform
D=imhmin(D,3);
L=watershed(D);
L(~imbin)=0;
WSmask=imfill(bwperim(L),'holes'); % Get mask seperating regions from watershed

%% Skeletonization

%     s=strel('diamond',1);
%     Iskel=bwmorph(imclose(imbinarize(imcomplement(imadjust(Iblur,[0.5 1]))),s),'skel',inf);
%     Ishrink=bwmorph(Iskel,'shrink',inf);
%     s=strel('diamond',1);%structure element
%     close=imdilate(Ishrink,s);
%     fill=imcomplement(imfill(close,'holes'));
%     overlay=rgb2gray(imcomplement(imoverlay(fill,close,'w'))); %true skeleton
%     thick=bwmorph(overlay,'thicken',10); %beefed up skeleton, distorts shape
%     thick=thick & imbin;
%     [B,L,N,A] = bwboundaries(bwperim(thick));

%% Polygonal approximation and ellipse fitting

[B,L] = bwboundaries(WSmask); % Boundary extraction from binarized, watershed image. B represents a structure containing the points along the perimeter of each boundary.
bubble_props=[]; % To store formatted ellipse properties
for i=1:length(B)
    boundary=B{i};
    points=[boundary(:,2)';boundary(:,1)']; % Formatting of boundary points in the form [x;y]
    
    [bp, bp_props]=polygonal_approximation(points,end_val); % Polygonization of points
    
    prop=ellipse_function(bp,bp_props,conc_sens); % Contour segmentation, segment grouping and parameter estimation are performed here
    bubble_props=[bubble_props,prop];
end

%% Save extracted ellipse properties from image

my_paramsf=zeros(length(props),6);
for i=1:length(props)
    temp=props(i);
    my_paramsf(i,:)=[temp.Centroid(1),temp.Centroid(2),temp.MajorAxisLength/2,temp.MinorAxisLength/2,deg2rad(-temp.Orientation),pi*(temp.MinorAxisLength/2)*(temp.MajorAxisLength/2)];
end
save(save_path,"output_params");


end
