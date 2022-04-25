function [ADD]=ADD(segment) %takes in a single segment [x;y], returns the ADD of segment and the single ellipse being fit to it
%SEGMENT GROUPING DONE ELSEWHERE

%if(size(segment,2)<10)
if(var(segment(1,:))<5 | var(segment(2,:))<5 | size(segment,2)<=5) 
    segment=[segment(1,find(segment(1,:))) ; segment(2,find(segment(2,:)))];
    mask=poly2mask(segment(1,:),segment(2,:),300,300);
        
    s = regionprops(mask,{...
        'Centroid',...
        'MajorAxisLength',...
        'MinorAxisLength',...abs(concave(i)-concave(i+1))+1abs(concave(i)-concave(i+1))+1
        'Orientation'})
    
     phi=0;Xc=0;Yc=0;a=0; b=0;amax=0;atemp=0;
     for k = 1:length(s) %strange behavior if no ellipses detected, could be a problem. need to only feed in combinations of segments which definitely can make ellipses
         atemp=s(k).MajorAxisLength/2;
         %if(atemp>=amax)
                a = s(k).MajorAxisLength/2;
                b = s(k).MinorAxisLength/2;
                Xc = s(k).Centroid(1);
                Yc = s(k).Centroid(2);
                phi = deg2rad(-s(k).Orientation);
                %amax=a;
     end
else
    [Xc, Yc, a, b, phi] = EllipseDirectFit(segment');
end
     %transform the first, middle, and last points of segment along the
     %fitted ellipse for compatison
    round(size(segment,2)/2)
    transformed=[[cos(phi) -sin(phi)];[sin(phi) cos(phi)]]*[[segment(1,1), segment(1,round(size(segment,2)/2)),segment(1,end)]-Xc;[segment(2,1), segment(2,round(size(segment,2)/2)),segment(2,end)]-Yc];
    %transformed=[[cos(phi) -sin(phi)];[sin(phi) cos(phi)]]*[segment(1,:)-Xc;segment(2,:)-Yc];%uses all points to
    %transform
    D=sqrt((transformed(1,:).^2)./(a^2)+(transformed(2,:).^2)./(b^2));
    ADD=abs(sum(sqrt((transformed(1,:).^2)+transformed(2,:).^2).*(1-1./D))/3); %divide by size(D,2) if using all points
    
end