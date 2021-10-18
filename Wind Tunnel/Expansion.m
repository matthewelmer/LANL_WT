function [ xList yList ] = Expansion(r, theta, n)
    rangeAngles = linspace(3*pi/2, 3*pi/2 + theta, n);
    xList = [];
    yList = [];
    for i = 1:length(rangeAngles)
        x = r*cos(rangeAngles(i));
        y = r*sin(rangeAngles(i)) + (1+r);
        xList(i) = x;
        yList(i) = y;
        plot(xList, yList, '-');
    end
    axis equal
end

    
