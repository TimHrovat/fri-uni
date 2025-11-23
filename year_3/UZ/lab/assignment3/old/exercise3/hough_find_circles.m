function A = hough_find_circles(I,r)
    [row,col] = size(I); 
    A = zeros(row,col); 

    for x=1:row
        for y=1:col
            if I(x,y)>0
                for t=0:360
                    a = round(x-r*cos(t * pi / 180));
                    b = round(y+r*sin(t * pi / 180));
                    if (a>0 && a<=row && b>0 && b<=col) 
                        A(a,b) = A(a,b)+1;
                    end
                end
            end
        end
    end