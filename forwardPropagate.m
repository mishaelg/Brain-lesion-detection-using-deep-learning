function [a,z] = forwardPropagate(input,weights,bias)
    a = zeros(size(bias));
    z = zeros(size(bias));
    for i=1:length(bias)
        z(i) = sum(input*weights(:,i)) + bias(i);
        a(i) = sigmoid(z(i));
    end
    a = single(a);
    z = single(z);
end