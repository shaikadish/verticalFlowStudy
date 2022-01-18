% Point placement algorithm

% seg -> input: Ellipse segment point is being added to
% centre -> output: Coordinates of ellipse centre with new point added
% A_final -> output: Major axis length with new point added
% B_final -> output: Minor axis length with new point added
% theta -> output: Ellipse angle of rotation after new point added
% new_point -> output: New point coordinates

function [centre, A_final, B_final,theta,new_point]=point_place(seg)

centre=0; A=0; B_final=0; theta=0; new_point=0;

%% Determination of opposite ellipse points
% p1 and p2 represent the coordinates of the opposite points

% Find index of median point
med_seg=[median(seg(1,:));median(seg(2,:))];
dmin=100;
for i=1:length(seg)
    d=norm(seg(:,i)-med_seg);
    if(d<dmin)
        dmin=d;
        index_med=i;
    end
end
if(index_med==1)
    index_med=2;
end

% If median point is on left half of ellipse, and therefore left side is
% longer
if(index_med<round(length(seg)/2))
    
    index_init=1;
    p1=seg(:,1);
    dmin=100;
    index=0;
    
    for i=index_med:length(seg)
        
        m1=(seg(2,1)-seg(2,i))/(seg(1,1)-seg(1,i));
        m3=(seg(2,1)-seg(2,2))/(seg(1,1)-seg(1,2));
        m2=(seg(2,i)-seg(2,i-1))/(seg(1,i)-seg(1,i-1));
        t2=abs(atan(m2)*180/pi);
        t1=abs(atan(m1)*180/pi);
        t3=abs(atan(m3)*180/pi);
        
        a1=abs(atan((m3-m1)/(1+m1*m3)));
        a2=abs(atan((m2-m1)/(1+m1*m2)));
        if(isnan(a1) & m2>5)
            a1=a2;
        end
        if(isnan(a2) & m1>5)
            a2=a1;
        end
        d=abs(a1-a2);
        if(d<dmin)
            dmin=d;
            index=i;
        end
    end
    
    % If median point on right side of ellipse, and right side longer
else
    p1=seg(:,end);
    index_init=length(seg);
    dmin=100;
    index=0;
    for i=2:index_med
        
        m1=(seg(2,end)-seg(2,i))/(seg(1,end)-seg(1,i));
        m2=(seg(2,end)-seg(2,end-1))/(seg(1,end)-seg(1,end-1));
        m3=(seg(2,i)-seg(2,i-1))/(seg(1,i)-seg(1,i-1));
        t2=abs(atan(m2)*180/pi);
        t1=abs(atan(m1)*180/pi);
        t3=abs(atan(m3)*180/pi);
        
        a1=abs(atan((m3-m1)/(1+m1*m3)));
        a2=abs(atan((m2-m1)/(1+m1*m2)));
        if(isnan(a1))
            a1=a2;
        end
        if(isnan(a2))
            a2=a1;
        end
        d=abs(a1-a2);
        if(d<dmin)
            dmin=d;
            index=i-1;
            
        end
    end
end

if(index==0)
    index=length(seg);
end
p2=seg(:,index);

if(abs(seg(1,1)-seg(1,end))<5)
    p1=seg(:,1); p2=seg(:,end);
end

%% Axis line generation

p=(p1(:)+p2(:))./2; % midpoint between start and end point
m=-1/((p2(2)-p1(2))/(p2(1)-p1(1))); % gradient of axis line
if(abs(m)==Inf | isnan(m))
    theta=pi/2;
    m=1000;
else
    theta=atan(m);
end

c=p(2)-p(1)*m % y intercept of axis line

% Parameters of perpendicular line to axis line which passes through the
% start and end point
m0=-1/m;
c0=p(2)-p(1)*m0;

%% Axis line seniority decision

sm_index=find(abs(seg(1,:)*m+c-seg(2,:))==min(abs(seg(1,:)*m+c-seg(2,:))));
sm_index=sm_index(1); % in cases where multiple detections
special_mid=seg(:,sm_index); % Special mid is the point which the axis line passes through from the existing points

% generate curve segment from which curature is measured
if(sm_index==length(seg))
    curve1=seg(:,sm_index-2); curve2=seg(:,sm_index)
elseif(sm_index==1)
    curve1=seg(:,sm_index); curve2=seg(:,sm_index+2)
else
    d1=norm(special_mid-seg(:,sm_index-1)); d2=norm(special_mid-seg(:,sm_index+1));d3=norm(special_mid-seg(:,1));d4=norm(special_mid-seg(:,end));
    ds=[d1,d2,d3,d4];
    if(abs(d1-d2)<=abs(d1-d4) & abs(d1-d2)<=abs(d2-d3) & abs(d1-d2)<=abs(d3-d4))
        curve1=seg(:,sm_index-1); curve2=seg(:,sm_index+1);
    elseif(abs(d1-d4)<=abs(d1-d2) & abs(d1-d4)<=abs(d2-d3) & abs(d1-d4)<=abs(d3-d4))
        curve1=seg(:,sm_index-1);curve2=seg(:,end);
    elseif(abs(d2-d3)<=abs(d1-d2) & abs(d2-d3)<=abs(d1-d4) & abs(d2-d3)<=abs(d3-d4))
        curve1=seg(:,1);curve2=seg(:,sm_index+1);
    else
        curve1=seg(:,sm_index-1); curve2=seg(:,sm_index+1);
    end
end
curvep=(curve1(:)+curve2(:))./2;
curve=norm(special_mid-curvep)/(2*norm(curvep-curve1)); % Curvature of curve segment. Used to determine axis line seniority

%% Major axis foci search
if(curve>=0.18)
    
    % Sample points along axis line
    if(abs(m)>20) % Special case when m is large
        line=[special_mid(1).*ones(1,401);special_mid(2)-10:0.05:special_mid(2)+10];
    else
        line=[special_mid(1)-80:0.1:special_mid(1)+80;m*(special_mid(1)-80:0.1:special_mid(1)+80)+c];
    end
    
    % Determine direction of axis line (depending on if curve is concave up
    % or down)
    dir=1;
    mean_seg=[mean(seg(1,:));mean(seg(2,:))];
    if((norm(line(:,1)-mean_seg)<norm(line(:,end)-mean_seg)))%& (abs(m)<10))
        line=flip(line,2); % if segment is concave up, F1 must be taken from lowest point
        dir=-1;
    end
    
    % Use second half of line for efficiency
    line=line(:,round(length(line)/2):end);
    
    varmin=100; % lowest variance observed
    v=100; % variance produced by current candidate foci
    
    % Loop through all candidate foci pairs
    for i=1:round(length(line))
        count=0;
        
        F1=line(:,i); % first candidate foci
        
        % Generation of perpendicular line to axis, passing through F1
        m0=-1/m;
        c0=F1(2)-F1(1)*m0;
        line_leng=abs(max(seg(1,:))-min(seg(1,:)));
        t=[F1(1)-line_leng:1:F1(1)+line_leng];
        perp_line=[t;m0*t+c0];
        
        % If perpendicular line is too far from segment, continue. For
        % efficiency
        dmin=100;
        for ii=1:length(perp_line)
            for j=1:length(seg)
                d=norm(perp_line(:,ii)-seg(:,j));
                if(d<dmin)
                    indexi=ii;
                    dmin=d;
                end
            end
        end
        
        if(dmin>2)
            continue
        end
        
        for j=i:length(line)
            F2=line(:,j); % Second candidate foci
            C=norm(F1-F2)/2;
            ds=[];
            
            for k=1:length(seg)
                D1=norm(F1-seg(:,k));
                D2=norm(F2-seg(:,k));
                ds(end+1)=D1+D2;
            end
            
            
            v=var(ds);
            a=mean(ds)/2'; % candidate major axis length
            if(varmin>v)
                
                varmin=v;
                
                % Assign output parameters
                centre=(F1(:)+F2(:))./2;
                A_final=max(a);
                B_final=sqrt(A^2-C^2);
                new_point=centre+(dir)*[cos(theta) -sin(theta);sin(theta) cos(theta)]*[A;0];
                
            end
        end
    end

%% Minor axis foci search

else
    
    % Length of axis line
    line_leng=abs(max(seg(1,:)-min(seg(1,:))))/2;
    if(line_leng<1)
        line_leng=2/line_leng;
    end
    
    % Sample points along axis line
    if(abs(m)>20) % Special case when m is large
        line=[special_mid(1).*ones(1,401);special_mid(2)-10:0.05:special_mid(2)+10];
    else
        line=[special_mid(1)-80:0.1:special_mid(1)+80;m*(special_mid(1)-80:0.1:special_mid(1)+80)+c];
    end
    
    % Determine direction of axis line (depending on if curve is concave up
    % or down)
    dir=1;
    mean_seg=[mean(seg(1,:));mean(seg(2,:))];
    if((norm(line(:,1)-mean_seg)<norm(line(:,end)-mean_seg))& (abs(m)<10)) % WOULD NEED TO CHANGE DIRECTION OF < FOR REAL IMAGES BECAUSE OF WAY Y AXIS IS INDEXED
        %if((special_mid(2)-mean(seg(2,:)))/(special_mid(1)-mean(seg(1,:)))*m<0 & abs(m)<10)
        line=flip(line,2); % if segment is concave up, F1 must be taken from lowest point
        dir=-1;
    elseif((mean(seg(2,:))<special_mid(2))&(abs(m)>10))
        line=flip(line,2);
        dir=-1;
    end
    
    % Only search half the axis line for efficiency
    line=line(:,round(length(line)/2):end);
    
    varmin=100; % Min variance
    v=100; % Current variance
    
    % Loop through all points along axis line
    for i=1:round(length(line))
        
        % generation of line perpendicular to axis line
        centre_perp=line(:,i);
        m0=-1/m;
        c0=centre_perp(2)-centre_perp(1)*m0;
        syms x y
        y=m0*x+c0;
        if(abs(m0)>20) % special case when large m0
            t=[centre_perp(1)-line_leng:0.1:centre_perp(1)+line_leng]; 
        else
            t=[centre_perp(1)-line_leng:1:centre_perp(1)+line_leng];
        end
        perp_line=[t;m0*t+c0];
        
        b=norm(centre_perp-special_mid); % candidate minor axis length         
        ds=[];
        % Loop through one half of the perpendicular line bisected by the
        % axis line
        for j=1:round(length(perp_line)/2)
            
            % Candidate foci pair
            F1=perp_line(:,j);
            F2=perp_line(:,end+1-j);
            
            C=norm(F1-F2)/2;

            for k=1:length(seg)
                D1=norm(F1-seg(:,k));
                D2=norm(F2-seg(:,k));
                ds(end+1)=D1+D2;
            end

            a=mean(ds)/2; % Candidate major axis length
            v=var(ds);
            if(varmin>v)
 
                varmin=v;
                
                % Assign output parameters
                A_final=a;
                B_final=B;
                centre=centre_perp;
                new_point=centre+dir*[cos(theta) -sin(theta);sin(theta) cos(theta)]*[B_final;0];
            end           
        end
    end   
end

end
