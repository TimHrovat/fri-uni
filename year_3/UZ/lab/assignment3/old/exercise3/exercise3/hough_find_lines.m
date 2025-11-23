function A = hough_find_lines(I, bins_rho, bins_theta)
    [r,c] = size(I);
    max_rho =  round(sqrt(r^2 + c^2));
    
    A = zeros(bins_rho,bins_theta);
    
    val_theta = (linspace(-90, 90, bins_theta) / 180) * pi;
    val_rho = linspace(-max_rho, max_rho, bins_rho);
    
    for i = 1:r
        for j = 1:c
            if I(i,j)>0
                rho = i * cos(val_theta) + j * sin(val_theta);
%                   find bin_rho that is closest to actual distance
                bin_rho = round(((rho + max_rho) / (2 * max_rho)) * length(val_rho));
                for l = 1:bins_theta
                    if bin_rho(l) > 0 && bin_rho(l) <= bins_rho
                        A(bin_rho(l), l) = A(bin_rho(l), l) + 1;
                    end
                end
            end
        end
    end