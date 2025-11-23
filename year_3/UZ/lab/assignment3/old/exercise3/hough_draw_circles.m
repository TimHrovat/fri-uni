function hough_draw_circles(I, A, rad, t)
% Draws the circles calculated by Hough transform to an image
%
% Input:
%  - I: image
%  - x: x coordinates
%  - y: y coordinates
%  - r: radii

x = [];
y = [];
r = [];
M = max(max(A));
threshold = M*t;

for k = 1:size(A,1)
    for j = 1:size(A,2)
        if A(k,j)>=threshold
            x = [x, k];
            y = [y, j];
            r = [r,rad];
        end
    end
end

circle_x = cos(linspace(0, 2*pi, 100));
circle_y = sin(linspace(0, 2*pi, 100));

imshow(I);
hold on;
for i = 1 : length(x)
    plot(circle_x * r(i) + x(i)-8, circle_y * r(i) + y(i)-5, 'g');
end
hold off;
