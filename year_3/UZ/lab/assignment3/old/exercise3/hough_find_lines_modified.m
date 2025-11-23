function A = hough_find_lines_modified(I,bins_rho, bins_theta)
    [r,c] = size(I);
    max_rho =  round(sqrt(r^2 + c^2));
    
    [Imag,Idir] = gradient_magnitude(I,2);
    Idir = Idir./2;
    
    A = zeros(bins_rho,bins_theta);
    
    val_theta = (linspace(-90, 90, bins_theta) / 180) * pi;
    val_rho = linspace(-max_rho, max_rho, bins_rho);
    
    for i = 1:r
        for j = 1:c
            if I(i,j)>0
                theta = Idir(i,j);
                rho = i * cos(theta) + j * sin(theta);
%                   find bin_rho that is closest to actual distance
                bin_rho = round(((rho + max_rho) / (2 * max_rho)) * length(val_rho));
                bin_theta = round((theta+pi/2)*length(val_theta)/pi);
                if bin_rho > 0 && bin_rho <= bins_rho
                    A(bin_rho, bin_theta) = A(bin_rho, bin_theta) + 1;  
                end
            end
        end
    end