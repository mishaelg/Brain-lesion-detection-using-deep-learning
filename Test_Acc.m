%% Testing set

[rowTest, colTest] = size(test);
outputTest = zeros(rowTest,1);
for t=1:rowTest
      [a1, z1] = forwardPropagate(test(t,:),whl,bhl);
      z2 = sum(a1'*wol) + bol;
      a2 = sigmoid(z2);
      outputTest(t) = a2;
end
outputTest=round(outputTest);
%figure;
accTest = zeros(rowTest,1);
for a = 1:rowTest
    accTest(a) = 100*(1-abs(outputTest(a)-ref_test(a)));
end
%accValidAvgVec = [accValidAvgVec , mean(accTest)];
b=nnz(accTest);
figure;
plot(accTest);
title('Accurcy for Test data set'); xlabel('Picture #'); ylabel('Accurcy [%]');