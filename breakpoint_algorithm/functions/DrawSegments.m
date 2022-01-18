function DrawSegments(s)

       t = linspace(0,2*pi,50);

        hold on
        for k = 1:length(s)
            a = s(k).MajorAxisLength/2;
            b = s(k).MinorAxisLength/2;
            Xc = s(k).Centroid(1);
            Yc = s(k).Centroid(2);
            phi = deg2rad(-s(k).Orientation);
            x = Xc + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
            y = Yc + a*cos(t)*sin(phi) + b*sin(t)*cos(phi);
            plot(x,y,'r','Linewidth',1);
        end
        
       hold off
end