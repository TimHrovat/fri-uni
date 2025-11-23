function [Ie] = findedges_hysteresis(I, sigma, low, high)
    x = 20;
    y = 20;
   	Image = padarray(I,[x y],'symmetric','both');
    Nkeep1 = size(Image,1);
    Nkeep2 = size(Image,2);

    [Imag,Idir] = gradient_magnitude(Image, sigma);
    ImageE = zeros(size(Imag));
    Imax = nonmaxima_suppression_line(Imag, Idir);
    ImageE(Imax>=low) = 1;
    ImageE(Imax>=high) = 2;
    Ie = ImageE(x:Nkeep1-y, x:Nkeep2-y);
    
    mask = bwlabel(Ie);
    
    figure(9); clf();
    imagesc(mask);
    
    label_max = max(mask(:));
    for i=1:label_max
        conn = Ie(mask==i);
        list = unique(conn);
        if (ismember(1,list) && ismember(2,list)) || ismember(2,list)
            Ie(mask==i) = 1;
        else
            Ie(mask==i) = 0;
        end
    end