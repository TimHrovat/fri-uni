%%
%Naloga 1b
impulse = zeros(25); 
impulse(13,13) = 255;
sigma = 6.0;
G = gauss(sigma);
D = gaussdx(sigma);


%%
%Naloga 1 b) a)
Ia = conv2(impulse,G);
Ia = conv2(Ia,G');

%%
%Naloga 1 b) b)
Ib = conv2(impulse,G);
Ib = conv2(Ib,D');

%%
%Naloga 1 b) c) 
Ic = conv2(impulse,D);
Ic = conv2(Ic,G');
%%
%Naloga 1 b) d)
Id = conv2(impulse,G');
Id = conv2(Id,D);
%%
%Naloga 1 b ) e)
Ie = conv2(impulse,D');
Ie = conv2(Ie,G);
figure(1); clf(); 
subplot(2,3,1); imagesc(impulse); title("impulse");
subplot(2,3,2); imagesc(Ia); 
subplot(2,3,3); imagesc(Ib); 
subplot(2,3,4); imagesc(Ic); 
subplot(2,3,5); imagesc(Id); 
subplot(2,3,6); imagesc(Ie); 
%%
%Naloga 1c
M = rgb2gray(imread("museum.jpg"));

sigma = 6;
[Mx,My] = image_derivatives(M,sigma);
[Mxx,Myy,Mxy] = image_derivatives2(M,sigma);
[Imag,Idir] = gradient_magnitude(M,sigma);

figure(); clf(); colormap("gray");
subplot(2,4,1); imagesc(M); 
subplot(2,4,2); imagesc(Mx); 
subplot(2,4,3); imagesc(My); 
subplot(2,4,4); imagesc(Imag); 
subplot(2,4,5); imagesc(Mxx); 
subplot(2,4,6); imagesc(Mxy); 
subplot(2,4,7); imagesc(Myy); 
subplot(2,4,8); imagesc(Idir);

%%
%Naloga 3a

bins_theta = 300; bins_rho = 300; % Resolution of the accumulator array
max_rho = 100; % Usually the diagonal of the image
val_theta = (linspace(-90, 90, bins_theta) / 180) * pi; % Values of theta are known
val_rho = linspace(-max_rho, max_rho, bins_rho);
X = [10,30,50,80];
Y = [10,60,20,90];
figure(5); clf();
for j = 1:4 
    A = zeros(bins_rho, bins_theta);
    x = X(j);
    y = Y(j);
    rho = x * cos(val_theta) + y * sin(val_theta); % compute rho for all thetas
    bin_rho = round(((rho + max_rho) / (2 * max_rho)) * length(val_rho)); % Compute bins for rho
    for i = 1:bins_theta % Go over all the points
        if bin_rho(i) > 0 && bin_rho(i) <= bins_rho % Mandatory out-of-bounds check
            A(bin_rho(i), i) = A(bin_rho(i), i) + 1; % Increment the accumulator cells
        end
    end
subplot(2,2,j); imagesc(A); % Display status of the accumulator
end

%%
%Naloga 3b

figure(6); clf();
oneLine = rgb2gray(imread("oneline.png"));

lineEdges = findedges_modified(oneLine,1,10);
lineHough = hough_find_lines(lineEdges,size(lineEdges,1),size(lineEdges,2));

subplot(1,2,1); imagesc(lineHough); title("oneline.png");

rectangle = rgb2gray(imread("rectangle.png"));
recEdges = findedges_modified(rectangle,1,10);
rec = hough_find_lines(recEdges,size(recEdges,1),size(recEdges,2));
subplot(1,2,2); imagesc(rec); title("rectangle.png");

%%
%Naloga 3c
figure(8); clf(); 
subplot(1,2,1); imagesc(rec); title("normal");
recMod = nonmaxima_suppression_box(rec);
subplot(1,2,2); imagesc(recMod); title("nonmaxima suppresion");


%%
%Naloge 3e
oneLine = rgb2gray(imread("oneline.png"));

lineEdges = findedges_modified(oneLine,1,5);
bins_rho = size(oneLine,1);
bins_theta = size(oneLine,2);
lineHough = hough_find_lines(lineEdges,bins_rho,bins_theta);
lineAcc = nonmaxima_suppression_box(lineHough);
hough_draw_lines(oneLine,lineAcc,bins_rho,bins_theta,0.8);






rectangle = rgb2gray(imread("rectangle.png"));

recEdges = findedges_modified(rectangle,1,5);
bins_rho = size(rectangle,1);
bins_theta = size(rectangle,2);
recHough = hough_find_lines(recEdges,bins_rho,bins_theta);
recAcc = nonmaxima_suppression_box(recHough);
hough_draw_lines(rectangle,recAcc,bins_rho,bins_theta,0.35);


%%
%Naloga 3f

bricksrgb = imread("bricks.jpg");
bricks = rgb2gray(bricksrgb);
pierrgb = imread("pier.jpg");
pier = rgb2gray(pierrgb);

bricksE = findedges_hysteresis(bricks,2,5,20);


pierE = findedges_hysteresis(pier,2,5,20);

b_bins_rho = size(bricksE,1);
b_bins_theta = size(bricksE,2);

bricksHough = hough_find_lines(bricksE,b_bins_rho,b_bins_theta);
bricksNMS = nonmaxima_suppression_box(bricksHough);
hough_draw_lines_top(bricksrgb,bricksNMS,b_bins_rho,b_bins_theta,10);



p_bins_rho = size(pierE,1);
p_bins_theta = size(pierE,2);

pierHough = hough_find_lines(pierE,p_bins_rho,p_bins_theta);
pierNMS = nonmaxima_suppression_box(pierHough);
hough_draw_lines_top(pierrgb,pierNMS,p_bins_rho,p_bins_theta,10);


figure(9); clf();
subplot(2,2,1); imagesc(bricksHough); title("BRICKS");
subplot(2,2,3); imagesc(bricksE);
subplot(2,2,2); imagesc(pierHough); title("PIER");
subplot(2,2,4); imagesc(pierE);

%%
%Naloga 3h

eclipsergb = imread("eclipse.jpg");
eclipse = rgb2gray(eclipsergb);

eclipseE = findedges_hysteresis(eclipse,2,15,20);
EHough = hough_find_circles(eclipseE,50);
E = nonmaxima_suppression_box(EHough);

figure(10); clf();
subplot(1,2,1); imagesc(eclipseE); title("find edges");
subplot(1,2,2); imagesc(EHough); title("find circles");


figure(11); clf();
hough_draw_circles(eclipsergb,E,50,0.8);
