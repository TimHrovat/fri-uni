function [Ix, Iy] = image_derivatives(I, sigma) 
    G = gauss(sigma);
    D = gaussdx(sigma);
    Ix = conv2(I,G');
    Ix = conv2(Ix,D);
    Iy = conv2(I,G);
    Iy = conv2(Iy,D');