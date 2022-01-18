% Function which groups ellipse segments together

% INPUTS:
% segments -> Uncgrouped ellipse contour segments

% OUTPUT:
% properties -> properties of each detected ellipse

function [new_segments]=segment_group(segments)

n=(size(segments(:,1),1)/2) % number of segments
padding=size(segments,2)*2; % get length of segments for zero padding later for returned matrix
new_segments=[]; % output, newly grouped segments
lastcheck=0; % if the last segment has been grouped, change to 1. if 0 at end, last segment is its own elipse
secondlastcheck=0;
skiplist=[]; % make list of segments which have been grouped so that they are skipped later on
ADDbest=100; % initial ADDbest

if((n==3)) % Special case when 3 segments
    
    % Group the smaller segment with the other segment which has the
    % lowest ADD when they are combined
    segmenti=[segments(1*2-1,find(segments(1*2-1,:))) ; segments(1*2,find(segments(2*1,:)))] %get segment without zero padding
    segmentj=[segments(2*2-1,find(segments(2*2-1,:))) ; segments(2*2,find(segments(2*2,:)))] %get segment without zero padding
    segmentk=[segments(3*2-1,find(segments(3*2-1,:))) ; segments(3*2,find(segments(2*3,:)))] %get segment without zero padding
    if(length(segmenti)<=length(segmentj) & length(segmenti)<=length(segmentk))
        segment0=[segmenti segmentj];
        segment1=[segmentk segmenti];
        if(ADD(segment0)<ADD(segment1))
            segment1=segmentk;
        else
            segment0=segmentj;
        end
    elseif(length(segmentj)<=length(segmenti) & length(segmentj)<=length(segmentk))
        segment0=[segmenti segmentj];
        segment1=[segmentj segmentk];
        if(ADD(segment0)<ADD(segment1))
            segment1=segmentk;
        else
            segment0=segmenti;
        end
    elseif(length(segmentk)<=length(segmenti) & length(segmentk)<=length(segmentj))
        segment0=[segmentk segmenti];
        segment1=[segmentj segmentk];
        if(ADD(segment0)<ADD(segment1))
            segment1=segmentj;
        else
            segment0=segmenti;
        end
    end
    segment0=unique(segment0','rows','stable')';
    segment1=unique(segment1','rows','stable')';
    new_segments=[[segment0 zeros(2,padding-size(segment0,2))];[segment1 zeros(2,padding-size(segment1,2))]];
    
elseif(n==2) % Special case when 2 segments
    
    % Determines wheather to group two segments, or leave them separate
    segmenti=[segments(1,find(segments(1,:))) ; segments(2,find(segments(2,:)))]
    segmenti=unique(segmenti','stable','rows')';
    ADDi=ADD([segmenti(1,:);segmenti(2,:)])
    segmentj=[segments(2*2-1,find(segments(2*2-1,:))) ; segments(2*2,find(segments(2*2,:)))]
    segmentj=unique(segmentj','stable','rows')';
    ADDj=ADD([segmentj(1,:);segmentj(2,:)])
    segmentij=[segmenti(1,:) segmentj(1,:); segmenti(2,:) segmentj(2,:)];
    ADDij=ADD([segmentij(1,:);segmentij(2,:)])
    
    if((ADDij<=(ADDi+ADDj))&(ADDij<=ADDbest)&(segmenti(1,1)~=100)) % Group descision
        best=[segmenti segmentj]
        ADDbest=ADDij
        best=[best zeros(2,padding-size(best,2))]
    else
        best=[[segmenti zeros(2,padding-size(segmenti,2))];[segmentj zeros(2,padding-size(segmentj,2))]]
    end
    
    new_segments=best
    
   
elseif(n==1) % Special case when 1 segment
    new_segments=segments;
    
else % Normal case, when there are more than 3 segments
     
    for i=1:n
        skiptemp=0;
        if(sum(ismember(skiplist,i))>0) % Skip grouped segments
            continue
        end
        
        segmenti=[segments(i*2-1,find(segments(i*2-1,:))) ; segments(i*2,find(segments(2*i,:)))] 
        ADDi=ADD([segmenti(1,:);segmenti(2,:)]);
        ADDbest=ADDi
        
        if(i~=1) %if i==1, dont go all the way to n
            best=[segmenti];% best (minimum) ADDij
            for j=i+1:n 
                if(sum(ismember(skiplist,j))>0)
                    continue
                end
                
                segmentj=[segments(j*2-1,find(segments(j*2-1,:))) ; segments(j*2,find(segments(2*j,:)))]
                ADDj=ADD([segmentj(1,:);segmentj(2,:)]);
                segmentij=[segmenti(1,:) segmentj(1,:); segmenti(2,:) segmentj(2,:)]
                ADDij=ADD([segmentij(1,:);segmentij(2,:)]);
                ADDij=coordinateInvariantEllipseFit(segmentij').geometricRMSE;
                ADDbest;

                if((ADDij<=(ADDi+ADDj))&(ADDij<=ADDbest)) % Group descision
                    best=[segmenti segmentj]
                    ADDbest=ADDij
                    skiptemp=j;
                    if(j==n)
                        lastcheck=1; % this means that the last segment has been grouped already
                    end
                    if(j==n-1)
                        secondlastcheck=1;
                    end
                end
            end
            
        else
            best=[segmenti];% best (minimum) ADDij
            for j=i+1:n % do not include last segment if using first segment
                segmentj=[segments(j*2-1,find(segments(j*2-1,:))) ; segments(j*2,find(segments(2*j,:)))]
                ADDj=ADD([segmentj(1,:);segmentj(2,:)])
                segmentij=[segmenti(1,:) segmentj(1,:); segmenti(2,:) segmentj(2,:)];
                ADDij=ADD([segmentij(1,:);segmentij(2,:)])

                if((ADDij<=(ADDi+ADDj))&(ADDij<=ADDbest)) % Group descision
                    best=[segmenti segmentj];
                    if(ADDij>0.01)
                        ADDbest=ADDij
                    end
                    skiptemp=j;
                    if(j==n)
                        lastcheck=1; % this means that the last segment has been grouped already
                    end
                    if(j==n-1)
                        secondlastcheck=1;
                    end
                    
                end
            end
        end
        
        if(skiptemp~=0)
            skiplist=[skiplist, skiptemp]; % Add grouped segments to skip list
        end
        
        best=[best zeros(2,padding-size(best,2))];
        new_segments=[new_segments;best]
    end
    
    if(secondlastcheck==0 & n>=3) %if second last segment not grouped, add to the new list of segment groups
        last=[segments(end-3,:);segments(end-2,:)];
        last=last(:,find(last(1,:)));
        last=[last zeros(2,padding-size(last,2))]
        new_segments=[new_segments;last]
        size(new_segments)
    end
    
    if(lastcheck==0 & n>=3) %if last segment not grouped, add to the new list of segment groups
        last=[segments(end-1,:);segments(end,:)];
        last=last(:,find(last(1,:)));
        last=[last zeros(2,padding-size(last,2))]
        
        new_segments=[new_segments;last];
    end
    
end

end