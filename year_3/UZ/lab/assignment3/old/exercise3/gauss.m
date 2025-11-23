function [g, x] = gauss(sigma)
    x = -round(3.0*sigma):round(3.0*sigma);
    g = 1/(sqrt(2*pi)*sigma) * exp(-(x.^2 ./ (2*sigma^2)));
    g = g / sum(g) ; % normalisation