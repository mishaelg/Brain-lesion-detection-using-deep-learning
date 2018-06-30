epoche = 1;
while epoche<1000
    
    %% 1 epoche of training
    perm=randperm(512);
    ref_train=ref_train(perm);
    train=train(perm,:);
    perm=randperm(155);
    ref_valid=ref_valid(perm);
    valid=valid(perm,:);
    perm=randperm(98);
    ref_test=ref_test(perm);
    test=test(perm,:);
    miniBatch = 8;
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
for m = 0:(row/miniBatch)-1
        for n=1:miniBatch
            [a1, z1] = forwardPropagate(train(m*miniBatch+n,:),whl,bhl);
            z2 = sum(a1'*wol) + bol;
            a2 = sigmoid(z2);
            output(m*miniBatch+n) = a2;
        end
end
    %figure;
    acc = zeros(row,1);
    loss= zeros(row,1);
    for a = 1:row
        acc(a) = 100*(1-abs(round(output(a))-ref_train(a)));
        loss(a)=100*(output(a)-ref_train(a))^2;
    end
    accAvgVec = [accAvgVec , mean(acc)];
    lossAvgVec= [lossAvgVec , mean(loss)];
    
    %% validation

    [rowValid, colValid] = size(valid);
    outputValid = zeros(rowValid,1);
    
    for v=1:rowValid
        [a1, z1] = forwardPropagate(valid(v,:),whl,bhl);
        z2 = sum(a1'*wol) + bol;
        a2 = sigmoid(z2);
        outputValid(v) = a2;
    end

%figure;
    accValid = zeros(rowValid,1);
    lossValid=zeros(rowValid,1);
    for a = 1:rowValid
        accValid(a) = 100*(1-abs(round(outputValid(a))-ref_valid(a)));
        lossValid(a)=100*(outputValid(a)-ref_valid(a))^2;
    end
    
    accValidAvgVec = [accValidAvgVec , mean(accValid)];
    lossValidAvgVec=[lossValidAvgVec,mean(lossValid)];
    epoche = epoche+1;
end
%% ploting the results
figure;
plot(accAvgVec);
title('Accurcy for training data set'); xlabel('Epoche #'); ylabel('Accurcy [%]');
figure;
plot(lossAvgVec);
title('Loss for training data set'); xlabel('Epoche #'); ylabel('Loss [%]');
figure;
plot(accValidAvgVec);
title('Accurcy for validation data set'); xlabel('Epoche #'); ylabel('Accurcy [%]');
figure;
plot(lossValidAvgVec);
title('Loss for training data set'); xlabel('Epoche #'); ylabel('lossValidAvgVec [%]');