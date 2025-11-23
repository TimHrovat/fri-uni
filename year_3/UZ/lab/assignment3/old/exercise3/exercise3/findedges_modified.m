function [Ie] = findedges_modified(I, sigma, theta)
    x = 20;
    y = 20;
   	Image = padarray(I,[x y],'symmetric','both');
    Nkeep1 = size(Image,1);
    Nkeep2 = size(Image,2);
    
    [Imag,Idir] = gradient_magnitude(Image, sigma);
    ImageE = zeros(size(Imag));
    Imax = nonmaxima_suppression_line(Imag, Idir);
    ImageE(Imax>=theta) = 1;
    
    Ie = ImageE(x:Nkeep1-y, x:Nkeep2-y);