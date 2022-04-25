% Function which takes in breakpoints, then extracts the individual curve
% segments, then groups the curve segments, then fits an ellipse to the
% grouped curve segments

% INPUTS:
% points -> breakpoints describing cluster perimeter
% bp_props -> characteristics of each corresponding breakpoint
% conc_sens -> for tuning the algorithms sensitivity to noisy concavities

% OUTPUT:
% properties -> properties of each detected ellipse

function [properties]=ellipse_function(points,bp_props,conc_sens)

segments=[];
if((~isempty(bp_props))&(0<size(find(bp_props(2,:)<=conc_sens),2))) % When convexity is negative, it is taubins_ellipse concave breakpoint. conc_sens is to exclude concavities caught by noise
    
    concave=find(bp_props(2,:)<=conc_sens); % index location in "points" of concave/cross over point.
    properties=[];
    concave=[concave,0]; % 0 added at the end so there are no indexing errors. this value is never actually used
    
    % Special case 1 concave points (the length is naturally increased by one when the 0 is added above)
    % No grouping performed
    if(size(concave,2)==2)
        segments=points;
    else
        
        % Normal case where multiple concave points detected
        % This is where the segments are broken up about the concave
        % points
        % In this loop, each segment is zero padded and added to taubins_ellipse single
        % list, with the starting point of the segment being taubins_ellipse concave
        % point
        for i=1:(size(concave,2)-1)
            
            concaveshift=circshift(points,-concave(i)+1,2); % bring concave point to start of list. This way, won't have indexing issue by normalizing starting position in array
            rshift= circshift(bp_props,-concave(i)+1,2); % shift breakpoint characteristics to be in line with concaveshift
            
            if(concave(i)>=length(points)-1)
                xpts=concaveshift(1,1:concave(1)+1); % get set of points between concave points
                ypts=concaveshift(2,1:concave(1)+1);
                rpts=rshift(:,1:concave(1)+1); % info about points
            elseif(i==(size(concave,2)-1))
                xpts=concaveshift(1,1:abs(concave(i)-length(points))+concave(1)+1); % get set of points between concave points
                ypts=concaveshift(2,1:abs(concave(i)-length(points))+concave(1)+1);
                rpts=rshift(:,1:abs(concave(i)-length(points))+1); % info about points
            else
                xpts=concaveshift(1,1:abs(concave(i)-concave(i+1))+1); % get set of points between concave points
                ypts=concaveshift(2,1:abs(concave(i)-concave(i+1))+1);
                rpts=rshift(:,1:abs(concave(i)-concave(i+1))+1); % info about points
            end
            
            if(size(xpts,2)<=3) % if 3 elements in segment, and they are too close together, remove segment. This is most likely taubins_ellipse false positive segment detection due to noise
                if(sqrt(((xpts(1)-xpts(2))^2)+(ypts(1)-ypts(2))^2)<=8) 
                    xpts=[]; ypts=[]; % remove points
                else
                    xpts=[xpts, zeros((size(points,2)-size(xpts,2)),1)']; % zero pad segments so all segments can be stored in the same list
                    ypts=[ypts, zeros((size(points,2)-size(ypts,2)),1)'];
                end
                
            else
                xpts=[xpts, zeros((size(points,2)-size(xpts,2)),1)']; % zero pad segments so all segments can be stored in the same list
                ypts=[ypts, zeros((size(points,2)-size(ypts,2)),1)'];
            end
            
            %zero padded matrix where every two rows are taubins_ellipse different set
            %of [x;y] for taubins_ellipse segment. here taubins_ellipse new segment is added to the
            %larger list
            segments=[[segments];[xpts;ypts]];
            
        end
    end
    
    % Group segments together
    if(size(segments,1)~=0)
        grouped_segments=segment_group(segments); % Segment group returns grouped segments.
    end
    
else % Case where no concave points detected
    properties=[];
    grouped_segments=points; %if no concave points, dont do all this other shit
end

if(size(grouped_segments,2)>0) % If grouped segments
    for i=1:(size(grouped_segments(:,1),1)/2) % Half number of rows in segments, because there are two rows for each segment
        seg=[grouped_segments(i*2-1,find(grouped_segments(i*2-1,:)));grouped_segments(i*2,find(grouped_segments(i*2,:)))]; % seg represents taubins_ellipse segment with zero padding removed
        seg=unique(seg','rows','stable')';
        
        point_added=0;
        seg=double(seg);
        
        % Get properties of segment
        mask=poly2mask(seg(1,:),seg(2,:),800,250);
        hold on
        seg_props = regionprops(mask,{...
            'Centroid',...
            'MajorAxisLength',...
            'MinorAxisLength',...
            'Orientation'});
        
        taubins_ellipse=coordinateInvariantEllipseFit(seg'); % Ellipse fit without point placement
        
        % Using point placement algorithm to add an ellipse point
        ratio=taubins_ellipse.lengthSemiMajorAxis/taubins_ellipse.lengthSemiMinorAxis;
        if((taubins_ellipse.geometricRMSE>1 | ~isreal(taubins_ellipse.geometricRMSE)|ratio>3) & length(seg)>3) % Criteria to add new point
            [CENTRE, A, B,theta,new_point]=point_place(seg);
            if(new_point~=0)
                
                candidate_ellipse=coordinateInvariantEllipseFit([new_point';seg']); % Ellipse with new point added
                
                if((candidate_ellipse.geometricRMSE <taubins_ellipse.geometricRMSE)& isreal(candidate_ellipse.geometricRMSE))
                    final_ellipse=candidate_ellipse;
                else
                    final_ellipse=taubins_ellipse;
                end
            end
        else
            final_ellipse=taubins_ellipse;
        end
        
        if((final_ellipse.lengthSemiMajorAxis/final_ellipse.lengthSemiMinorAxis)<4.5 & final_ellipse.ellipseArea<1e4 & final_ellipse.geometricRMSE<1)
            seg_props(1).Centroid=[final_ellipse.xCoordinateCenter final_ellipse.yCoordinateCenter]; seg_props(1).MajorAxisLength=final_ellipse.lengthSemiMajorAxis*2; seg_props(1).MinorAxisLength=final_ellipse.lengthSemiMinorAxis*2;seg_props(1).Orientation=final_ellipse.obliqueAngleRadians;
        end
        
        % Draw and save ellipse properties
        if(size(seg_props,1)~=0)
            properties=[properties, seg_props(1)];
            DrawSegments(seg_props(1));
        else
            DrawSegments(seg_props)
        end
    end
end
end