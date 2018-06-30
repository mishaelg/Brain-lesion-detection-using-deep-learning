function [w1,w2,b1,b2] = backPropagate(w1,w2,b1,b2,z1,z2,a1,input,output,ref,eta)
    sum = 0;
    for i=1:length(output)
        sum = sum+(output(i)-ref(i));
    end
    dL = single(sigmoidTag(z2)*(sum/16)); %output layer error
    
    % output layer weight and bias update
    for i=1:length(w2)
        w2(i) = single(w2(i)-(eta/16)*dL*a1(i));
    end
    b2 = single(b2-(eta/16)*dL);
    
    dl = single(zeros(size(w2))); %hidden layer error
    for i=1:length(w2)
        dl(i) = single(dL*w2(i)*sigmoidTag(z1(i)));
    end
    
    % hidden layer weight and bias update
    for i=1:length(b1)
        b1(i) = single(b1(i)-(eta/16)*dl(i));
        for j=1:length(w1)
            w1(j,i)= single(w1(j,i)-(eta/16)*dl(i)*input(i));
        end   
    end   
end