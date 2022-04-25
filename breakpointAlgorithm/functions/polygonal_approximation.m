% Polygonal Aproxximation using Carmona-Poyato method

% points -> input: set of boundary points to be approximated
% endvalue -> input: A tunable parameter relating to the stopping condition
%                    of the algorithm. Larger endvalue, fewer breakpoints.
% breakpoints -> output: The points describing the polygonal approximation
%                        of "points".
% properties -> output: corresponding characteristics of each breakpoint. Of
%               the form properties=[breakpoint curvature; breakpoint convexity]

function [breakpoints, properties]=polygonal_approximation(points,endvalue)

breakpoints=points;
Dti=0.5;
ri=0;
Dmax=0;
size(breakpoints,2);
properties=[]
while((Dmax<=endvalue)&(size(breakpoints,2)>=3)) % reduce number of points until end condition is met where Dmax<=endvalue, or there are only 3 remaining breakpoints
    I=breakpoints; %list with index i
    
    if(size(I)<=3) % if less than 3 breakpoints, break
        break
    end
    
    J=circshift(breakpoints,-1,2); %make seperate list of points all shifted by one to the left. this is a list of index i-1
    K=circshift(breakpoints,-2,2); % A list of index i+1
    breakpoints=[];
    
    % Each loop starts at breakpoint with maximum curvature in previous
    Dmax=0; % measure of maximum curvature to find next loops starting position
    NextIP=[]; %next initial point, IP of next iteration
    properties=[];% top row is a measure equivalend of curvature,from Teh and Chin, rik=dik/lik=(distance between pi and chord pi-kpi+k)/(length of chord lik)
    
    for i=1:length(J)
        
        Xk=K(1,1); Yk=K(2,1);
        Xj=J(1,1); Yj=J(2,1);
        Xi=I(1,1); Yi=I(2,1);
        
        Numerator=((Xj-Xk)*(Yi-Yk)-(Yj-Yk)*(Xi-Xk))^2;
        Denominator=(Xi-Xk)^2+(Yi-Yk)^2;
        D=sqrt(Numerator/Denominator);
        
        if(D>=Dmax)
            Dmax=D;
            NextIP=[Xj;Yj];
        end
        
        if(D>Dti)
            breakpoints=[breakpoints, [Xj;Yj]]; %store breakpoints
            I=circshift(I,-1,2); %Why use circshift instead of straight up indexing? because removing elements, so indexing could be wack
            J=circshift(J,-1,2);
            K=circshift(K,-1,2);
            
        else
            deltaI=abs(sqrt(Denominator)-sqrt((Xi-Xj)^2+(Yi-Yj)^2)-sqrt((Xj-Xk)^2+(Yj-Yk)^2)); %change in  boundary length when point removed
            
            if(size(I,2)>2)
                I(:,2)=[]; % remove point in I. This point reduction contributes to the final reduction in points from "points" to "breakpoints"
                J=circshift(J,-1,2);
                K=circshift(K,-1,2);
            end
        end
    end
    
    IndexNextIP=find(I(1,:)==NextIP(1) & I(2,:)==NextIP(2)); %get index of next starting position
    
    if(IndexNextIP>1)
        breakpoints=circshift(I,-IndexNextIP(1)+1,2); %shift breakpoints so starting in next position for next iteration
    else
        breakpoints=I;
    end
    
    I2=circshift(breakpoints,1,2); %for calculating convexities and r
    K2=circshift(breakpoints,-1,2);
    convex=[];
    for i=1:size(breakpoints,2)
        
        Xk=K2(1,i); Yk=K2(2,i);
        Xj=breakpoints(1,i); Yj=breakpoints(2,i);
        Xi=I2(1,i); Yi=I2(2,i);
        
        %calculate convex points, negative if convex
        P1=[Xi;Yi];P2=[Xj;Yj];P3=[Xk;Yk];
        D1=P2-P1;D2=P3-P2;
        convex=[convex,D1(1)*D2(2)-D1(2)*D2(1)];
        
        %calculate curvature
        Numerator=((Xj-Xk)*(Yi-Yk)-(Yj-Yk)*(Xi-Xk))^2;
        Denominator=(Xi-Xk)^2+(Yi-Yk)^2;
        D=sqrt(Numerator/Denominator);
        properties=[properties, [D/sqrt((Xi-Xk)^2+(Yi-Yk)^2)]]; %calculate curvature for each point
    end
    
    properties=[properties;convex];% proerties includes curvature and convexity information
    Dti=Dti+0.5;
end

end

