function B = nonmaxima_suppression_box(A)
    B = A;
    for k=1:size(A,1)
        for l=1:size(A,2)
            podA = A(max(1,k-1):min(size(A,1),k+1), max(1,l-1):min(size(A,2),l+1));
            maxA = max(podA,[],'all');
            if B(k,l)<maxA
                B(k,l)=0;
            end
        end
    end