[RGB,map] = imread('image_HW5.jpg');
figure(1),
imshow(RGB);
%%
p = impixel(RGB,321,481);  %%Limit of the image
data(1,:) = [1 1 impixel(RGB,321,481)];
%%
x = 0;
for i = 1:321
    for j = 1:481
        x= x+1;
        data(x,:) = [i 482-j impixel(RGB,i,j)];
    end
end
%%
N = normalize(data);

%%
AIC = zeros(1,6);
GMModels = cell(1,6);
options = statset('MaxIter',500);
for k = 1:6
    GMModels{k} = fitgmdist(N,k,'Options',options,'CovarianceType','diagonal');
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents] = min(AIC);
numComponents;

BestModel = GMModels{numComponents};

%%
prior = BestModel.ComponentProportion;
mu = BestModel.mu;
sigma = BestModel.Sigma;

%%
for i = 1:154401    %154401 samples
    for j = 1:6     % 6 componnents
        Likeli(i,j) = mvnpdf(N(i,:),mu(j,:),sigma(:,:,j));
    end
end

%%
for i = 1:154401
    y = max(Likeli(i,:));
    if(y == Likeli(i,1))
        labels(i) = 1;
    elseif (y == Likeli(i,2))
        labels(i) = 2;
    elseif (y == Likeli(i,3))
        labels(i) = 3;
    elseif (y == Likeli(i,4))
        labels(i) = 4;
    elseif (y == Likeli(i,5))
        labels(i) = 5;
    elseif (y == Likeli(i,6))
        labels(i) = 6;
    end
end

%%
indl1 = find(labels == 1);
indl2 = find(labels == 2);
indl3 = find(labels == 3);
indl4 = find(labels == 4);
indl5 = find(labels == 5);
indl6 = find(labels == 6);

%%
figure (2),
plot(N(indl1,1),N(indl1,2),'m.'), hold on, 
plot(N(indl2,1),N(indl2,2),'g.'), hold on, 
plot(N(indl3,1),N(indl3,2),'b.'), hold on, 
plot(N(indl4,1),N(indl4,2),'y.'), hold on, 
plot(N(indl5,1),N(indl5,2),'r.'), hold on, 
plot(N(indl6,1),N(indl6,2),'c.'), hold on, 