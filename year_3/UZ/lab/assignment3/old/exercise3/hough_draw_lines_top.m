function hough_draw_lines_top(I, A, bins_rho, bins_theta,n)
% Draws the lines calculated by Hough transform to an image
%
% Input:
%  - I: image
%  - rho: displacement parameter
%  - theta: angle parameter

r = size(A,1);
c = size(A,2);
val_theta = (linspace(-90, 90, bins_theta) / 180) * pi;
val_rho = linspace(-round(sqrt(r^2 + c^2)), round(sqrt(r^2 + c^2)), bins_rho);

sorted = sort(A(:),'descend');
threshold = sorted(n+1);

rho = [];
theta = [];

for i = 1:r
    for j = 1:c
        if A(i,j)>=threshold
            rho = [rho, val_rho(i)];
            theta = [theta, val_theta(j)];
        end
    end
end

h = size(I, 1);
w = size(I, 2);

figure();
imshow(I);
hold on;
for i = 1 : length(theta)
    if abs(theta(i)) > pi/4
        x1 = 0;
        y1 = (rho(i) - x1 .* cos(theta(i))) ./ sin(theta(i))-12;
        x2 = h;
        y2 = (rho(i) - x2 .* cos(theta(i))) ./ sin(theta(i))-12;
    else
        y1 = 0;
        x1 = (rho(i) - y1 .* sin(theta(i))) ./ cos(theta(i))-5;
        y2 = w;
        x2 = (rho(i) - y2 .* sin(theta(i))) ./ cos(theta(i))-5;
    end
    plot([y1, y2], [x1, x2], 'g');
end
hold off;
