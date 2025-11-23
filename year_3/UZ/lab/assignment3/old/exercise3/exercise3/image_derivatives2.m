function [Ixx, Iyy, Ixy] = image_derivatives2(I, sigma) 
    G = gauss(sigma);
    D = gaussdx(sigma);
    [Ix,Iy] = image_derivatives(I,sigma);
    
    Ixx = conv2(Ix,G');
    Ixx = conv2(Ixx,D);
    
    Iyy = conv2(Iy,G);
    Iyy = conv2(Iyy,D');
    
    Ixy = conv2(Ixx,G);
    Ixy = conv2(Ixy,D');