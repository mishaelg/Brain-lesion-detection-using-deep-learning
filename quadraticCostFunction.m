function q = quadraticCostFunction(networkOutput,ref)
    sum = 0;
    for i = 1:length(ref)
        sum = sum + (networkOutput(i)-ref(i))^2;
    end
    q = sum/(2*length(ref));
    q = single(q);
end