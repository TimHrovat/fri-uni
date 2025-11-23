function [Ie] = findedges(I, sigma, theta)
    [Imag,Idir] = gradient_magnitude(I, sigma);
    Ie = zeros(size(Imag));
    Ie(Imag>=theta) = 1;