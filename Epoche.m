%% 1 epoche of training
miniBatch = 16;
[row,col] = size(train);
z1 = single(zeros(hiddenLayerSize,1));
a1 = single(zeros(hiddenLayerSize,1));
z2=single(0);
a2=single(0);
output = single(zeros(row,1));
qcf = single(zeros(row/miniBatch,1));

for m = 0:(row/miniBatch)-1
    for n=1:miniBatch
        [a1, z1] = forwardPropagate(train(m*miniBatch+n,:),whl,bhl);
        z2 = sum(a1'*wol) + bol;
        a2 = sigmoid(z2);
%         [a2, z2] = forwardPropagate(a1',wol,bol);
        output(m*miniBatch+n) = a2;
    end
    qcf(m+1) = quadraticCostFunction(output(m*miniBatch+1:m*miniBatch+n),ref_train(m*miniBatch+1:m*miniBatch+n));
    [whl,wol,bhl,bol] = backPropagate(whl,wol,bhl,bol,z1,z2,a1,bpInput,output(m*miniBatch+1:m*miniBatch+n),ref_train(m*miniBatch+1:m*miniBatch+n),eta);
end

figure;
acc = zeros(row,1);
for a = 1:row
    acc(a) = 100*(1-abs(output(a)-ref_train(a)));
end
plot(acc); title('Accurcy for train epoche #'); xlabel('mini batch number'); ylabel('Accurcy [%]');
%% validation

[rowValid, colValid] = size(valid);
outputValid = zeros(rowValid,1);
for v=1:length(rowValid)
        [a1, z1] = forwardPropagate(valid(v,:),whl,bhl);
        z2 = sum(a1'*wol) + bol;
        a2 = sigmoid(z2);
        outputValid(v) = a2;
end

figure;
accValid = zeros(rowValid,1);
for a = 1:rowValid
    accValid(a) = 100*(1-abs(output(a)-ref_train(a)));
end
plot(accValid); title('Accurcy for validation after epoche #'); xlabel('frame index'); ylabel('Accurcy [%]');



