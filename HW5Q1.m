%% Parameters
P0 = 0.5; p1 = 0.5; %prior
r0 = 2; r1 = 4; %constant
mu = [0;0]; C = [1 0;0 1]; %normal distribution
%% gernerate data
[x_1k,labels_1k] = generateDataAQ1(1000); %sample
[x_10k,labels_10k] = generateDataAQ1(10000); %testing
labels_1k = labels_1k';
labels_10k = labels_10k';
x_1k_new = x_1k';
x_10k_new = x_10k';

%% SVM classifier
for i = 1:1000
    if labels_1k(i) == 1
        labels_1k_new(i) = 1;
    else
        labels_1k_new(i) = -1;
    end  
end
labels_1k_new = labels_1k_new';

for i = 1:10000
    if labels_10k(i) == 1
        labels_10k_new(i) = 1;
    else
        labels_10k_new(i) = -1;
    end  
end
labels_10k_new = labels_10k_new';
%%
data3 = x_1k_new;
theclass = labels_1k_new;
%%
%Train the SVM Classifier
cl = fitcsvm(data3,theclass,'KernelFunction','rbf',...
    'BoxConstraint',4,'ClassNames',[-1,1]);

% Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);

% Plot the data and the decision boundary
figure;
% h(1:2) = gscatter(x_10k_new(:,1),x_10k_new(:,2),theclass,'rb','.');
hold on
% h(3) = plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off


%%
prediction_SVM = predict(cl,x_10k_new);

for i = 1:10000
    if prediction_SVM(i) >= 0
        prediction_SVM(i) = 1;
    else
        prediction_SVM(i) = -1;
    end
end

error = 0;
for i = 1:10000
    if prediction_SVM(i) ~= labels_10k_new(i)
        error = error + 1;
    end
end


%% MLP
y1 = trainedModel1.predictFcn(x_10k);
for i = 1:10000
    if y1(i) > 0.5
        z1(i) = 1;
    elseif y1(i) <= 0.5
        z1(i) = 0;
    end
end

error1 = 0; p0r0=0; p0r1=0; p1r0=0; p1r1=0;
for i = 1:10000
    if z1(i)==0
        if labels_10k(i) == 0
            p0r0 = p0r0+1;
        else
            p0r1 = p0r1 +1;
        end
    else
        if labels_10k(i) == 0
            p1r0 = p1r0+1;
        else
            p1r1 = p1r1 +1;
        end
    end
end

%%
function [x,labels] = generateDataAQ1(N)
classPriors = [0.5,0.5]; %prior
r0 = 2; r1 = 4; %constant
labels = (rand(1,N) >= classPriors(1));
mu = [0;0]; C = [1 0;0 1];
sita = rand(N,1);
for i = 1:N
    ind(i)=i;
    sita (i) = (sita(i)*2-1)*pi;
end
normal(:,ind) = mvnrnd(mu,C,N)';

for l = 0:1
    indl = find(labels==l);
    if l == 0
        comp(1,indl) = cos(sita(indl)) *r0;
        comp(2,indl) = sin(sita(indl)) *r0;
        x(:,indl) = comp(:,indl)+normal(:,indl);
        plot(x(1,indl),x(2,indl),'ro'), hold on,
    elseif l==1
        comp(1,indl) = cos(sita(indl)) *r1;
        comp(2,indl) = sin(sita(indl)) *r1;
        x(:,indl) = comp(:,indl)+normal(:,indl);
        plot(x(1,indl),x(2,indl),'b+'), hold on,
    end
end
end
