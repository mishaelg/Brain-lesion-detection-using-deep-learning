% Train QCF plot
[rowt,colt] = size(train);
outputt = zeros(rowt,1);
quadCostTrain = zeros(rowt,1);

for m = 1:rowt
    [a1t, z1t] = forwardPropagate(train(m,:),whl,bhl);
    z2t = sum(a1t'*wol) + bol;
    a2t = sigmoid(z2t);
    outputt(m) = a2t;
    quadCostTrain(m) = quadraticCostFunction(outputt(m),ref_train(m));
end

figure;
plot(quadCostTrain);
title('Quadratic cost function for training data set'); xlabel('image #'); ylabel('QCF');

%% Validation QCF plot
[rowv,colv] = size(valid);
outputv = zeros(rowv,1);
quadCostValid = zeros(rowv,1);

for n = 1:rowv
    [a1v, z1v] = forwardPropagate(valid(n,:),whl,bhl);
    z2v = sum(a1v'*wol) + bol;
    a2v = sigmoid(z2v);
    outputv(n) = a2v;
    quadCostValid(n) = quadraticCostFunction(outputv(n),ref_valid(n));
end

figure;
plot(quadCostValid);
title('Quadratic cost function for validation data set'); xlabel('image #'); ylabel('QCF');