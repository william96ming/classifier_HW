%%
%Parameters for two classes 
p0 = 0.6;  %Prior for label 0 
p1 = 0.4;  %Prior for label 1 
w1 = 0.5;   %Weight for label 0 condition 1 
w2 = 0.5;   %Weight for label 0 condition 2 
m01 = [5;0]; C01 = [4,0;0,2]; 
m02 = [0;4]; C02 = [1,0;0,3]; 
m1 = [3;2]; C1 = [2,0;0,2]; 
%%
N=100;
[x_100,labels_100] = generateDataA1Q1(N);
N=1000;
[x_1K,labels_1K] = generateDataA1Q1(N);
N=10000;
[x_10K,labels_10K] = generateDataA1Q1(N);
N=20000;
[x_20K,labels_20K] = generateDataA1Q1(N);
%% PartA 
%% Likelihood
for i=1:20000
Likelihood_L0(:,i)=w1*mvnpdf(x_20K(:,i),m01,C01)+w2*mvnpdf(x_20K(:,i),m02,C02);
Likelihood_L1(:,i)= mvnpdf(x_20K(:,i),m1,C1);
end

%% Discriminant function & decision
gx_0 = Likelihood_L0 * p0;
gx_1 = Likelihood_L1 * p1;
for i = 1:20000
    if(gx_0(:,i)>=gx_1(:,i))
        decision_1(:,i) = 0;
    else
        decision_1(:,i)=1;
    end
end
%% #of 0 & 1 in sample
N0 = length(find(labels_20K==0)); 
N1 = length(find(labels_20K==1)); 
%% min-Perror classifier
    TP = sum(decision_1==1 & labels_20K==1);
    FN = sum(decision_1==0 & labels_20K==1);
    FP = sum(decision_1==1 & labels_20K==0);
    TN = sum(decision_1==0 & labels_20K==0);
    X_ROC_classifier = FP/(TN+FP);
    Y_ROC_classifier = TP/(TP +FN);
    Pfa = sum(decision_1==1 & labels_20K==0)/N0; 
    Pcd = sum(decision_1==1 & labels_20K==1)/N1; 
    P_error = Pfa*p0+(1-Pcd)*p1; 
%% 
for ROC_parameter = 0:1:2000
    for i = 1:20000
        if((Likelihood_L1(:,i)/Likelihood_L0(:,i)) > (ROC_parameter/100))
            decision_ROC(:,i) = 1;
        else
            decision_ROC(:,i) = 0;
        end
    end
    TP = sum(decision_ROC==1 & labels_20K==1);
    FN = sum(decision_ROC==0 & labels_20K==1);
    FP = sum(decision_ROC==1 & labels_20K==0);
    TN = sum(decision_ROC==0 & labels_20K==0);
    X_ROC(:,(ROC_parameter+1)) = FP/(TN+FP);
    Y_ROC(:,(ROC_parameter+1)) = TP/(TP +FN);
     Pfa_ROC(:,(ROC_parameter+1)) = sum(decision_ROC==1 & labels_20K==0)/N0; 
     Pcd_ROC(:,(ROC_parameter+1)) = sum(decision_ROC==1 & labels_20K==1)/N1;
     P_error_ROC(:,(ROC_parameter+1)) = Pfa_ROC(:,(ROC_parameter+1))*p0+(1-Pcd_ROC(:,(ROC_parameter+1)))*p1; 
end
%% minimal P-error
P_error_mini = min(P_error_ROC );
temp = find(P_error_ROC == P_error_mini);
%% ROC curve
plot(X_ROC,Y_ROC,'-',X_ROC_classifier,Y_ROC_classifier,'o',X_ROC(:,181),Y_ROC(:,181),'x');
title('ROC Curve'); 
legend('ROC Curve','mini-classifier','estimated-min'); 
xlabel('FP rate'); 
ylabel('TP rate'); 
%%
%% Part B
%%

%% prior
N = 100;
number_L0_N = sum(labels_100 == 0);
prior_L0_N = number_L0_N / N;
number_L1_N = N - number_L0_N;
prior_L1_N = 1 - prior_L0_N;
%% conditional pdfs
sum_L1 = 0;
sum_L0 = 0;
j = 1;
clear element_L0;
%seperate elements with label L0,L1
for i = 1:N
    if labels_100(:,i) == 1
        sum_L1 = sum_L1 + x_100(:,i);
    else
        element_L0(:,j) = x_100(:,i);
        j = j + 1;
    end
end
mean_L1 = sum_L1 /number_L1_N;
fst_element_L1 = 0;
srd_element_L1 = 0;
td_element_L1 = 0;
temp = 0;
%find mean and covariance of class L1
for i = 1:N
    if labels_100(:,i) == 1
        fst_element_L1 = fst_element_L1 + (x_100(1,i) - mean_L1(1,1))^2;
        srd_element_L1 = srd_element_L1 + (x_100(2,i) - mean_L1(2,1))^2; 
        td_element_L1 = td_element_L1 + (x_100(1,i) - mean_L1(1,1))*(x_100(2,i) - mean_L1(2,1));
    end
end
sigma_L1_1 = fst_element_L1/number_L1_N;
sigma_L1_2 = srd_element_L1/number_L1_N;
sigma_L1_3 = td_element_L1/number_L1_N;
%k-means
[idx,C]=kmeans((element_L0)',2);
%find mean and covariance of class L0
number_01=length(find(idx == 1));
number_02=length(find(idx == 2));
number_L0=length(idx);
fst_element_01 = 0;
srd_element_01 = 0;
td_element_01 = 0;
fst_element_02 = 0;
srd_element_02 = 0;
td_element_02 = 0;

for i = 1:number_L0
    if idx(i) == 1
        fst_element_01 = fst_element_01+ (element_L0(1,i) - C(1,1))^2;
        srd_element_01 = srd_element_01 + (element_L0(2,i) - C(2,1))^2;
        td_element_01 = td_element_01 + (element_L0(1,i) - C(1,1)) * (element_L0(2,i) - C(2,1));
    end
    if idx(i) == 2
        fst_element_02 = fst_element_02+ (element_L0(1,i) - C(1,2))^2;
        srd_element_02 = srd_element_02 + (element_L0(2,i) - C(2,2))^2;
        td_element_02 = td_element_02 + (element_L0(1,i) - C(1,2)) * (element_L0(2,i) - C(2,2));
    end
end
sigma_01_1 = fst_element_01/number_01;
sigma_01_2 = srd_element_01/number_01;
sigma_01_3 = td_element_01/number_01;
sigma_02_1 = fst_element_02/number_02;
sigma_02_2 = srd_element_02/number_02;
sigma_02_3 = td_element_02/number_02;
%% plot after K-means
plot(element_L0(1,idx==1),element_L0(2,idx==1),'r.','MarkerSize',12)
hold on
plot(element_L0(1,idx==2),element_L0(2,idx==2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3) ;
title('clusters and center point in class L=0, sperate by k-means'); 
    
%% new likelihood training by N samples
Sigma01_N = [sigma_01_1 sigma_01_3;sigma_01_3 sigma_01_2];
Sigma02_N = [sigma_02_1 sigma_02_3;sigma_02_3 sigma_02_2];
sigma1_N = [sigma_L1_1 sigma_L1_3;sigma_L1_3 sigma_L1_2];
m01_N = C(:,1);
m02_N = C(:,2);
m1_N = mean_L1;
p0 = prior_L0_N;
p1 = prior_L1_N;

%% Likelihood
for i=1:20000
Likelihood_L0(:,i)=w1*mvnpdf(x_20K(:,i),m01_N,Sigma01_N)+w2*mvnpdf(x_20K(:,i),m02_N,Sigma02_N);
Likelihood_L1(:,i)= mvnpdf(x_20K(:,i),m1_N,sigma1_N);
end

%% Discriminant function & decision
gx_0 = Likelihood_L0 * p0;
gx_1 = Likelihood_L1 * p1;
for i = 1:20000
    if(gx_0(:,i)>=gx_1(:,i))
        decision_1(:,i) = 0;
    else
        decision_1(:,i)=1;
    end
end
%% #of 0 & 1 in sample
N0 = length(find(labels_20K==0)); 
N1 = length(find(labels_20K==1)); 
%% min-Perror classifier
    TP = sum(decision_1==1 & labels_20K==1);
    FN = sum(decision_1==0 & labels_20K==1);
    FP = sum(decision_1==1 & labels_20K==0);
    TN = sum(decision_1==0 & labels_20K==0);
    X_ROC_classifier = FP/(TN+FP);
    Y_ROC_classifier = TP/(TP +FN);
    Pfa = sum(decision_1==1 & labels_20K==0)/N0; 
    Pcd = sum(decision_1==1 & labels_20K==1)/N1; 
    P_error = Pfa*p0+(1-Pcd)*p1; 
%% 
for ROC_parameter = 0:1:2000
    for i = 1:20000
        if((Likelihood_L1(:,i)/Likelihood_L0(:,i)) > (ROC_parameter/100))
            decision_ROC(:,i) = 1;
        else
            decision_ROC(:,i) = 0;
        end
    end
    TP = sum(decision_ROC==1 & labels_20K==1);
    FN = sum(decision_ROC==0 & labels_20K==1);
    FP = sum(decision_ROC==1 & labels_20K==0);
    TN = sum(decision_ROC==0 & labels_20K==0);
    X_ROC(:,(ROC_parameter+1)) = FP/(TN+FP);
    Y_ROC(:,(ROC_parameter+1)) = TP/(TP +FN);
     Pfa_ROC(:,(ROC_parameter+1)) = sum(decision_ROC==1 & labels_20K==0)/N0; 
     Pcd_ROC(:,(ROC_parameter+1)) = sum(decision_ROC==1 & labels_20K==1)/N1;
     P_error_ROC(:,(ROC_parameter+1)) = Pfa_ROC(:,(ROC_parameter+1))*p0+(1-Pcd_ROC(:,(ROC_parameter+1)))*p1; 
end
%% minimal P-error
P_error_mini = min(P_error_ROC );
temp = find(P_error_ROC == P_error_mini);
%% ROC curve
plot(X_ROC,Y_ROC,'-',X_ROC_classifier,Y_ROC_classifier,'o',X_ROC(:,181),Y_ROC(:,181),'x');
title('ROC Curve with 100 training data'); 
legend('ROC Curve','from training data','estimated-min'); 
xlabel('FP rate'); 
ylabel('TP rate'); 

%% Part 3
%% linear logistic
data_N_t = x_100';
labels_N_t = labels_100';
labels_N_t = labels_N_t+1;
B_100 = mnrfit(data_N_t,labels_N_t);

%% validation

x_test_20K = [ones(20000,1) (x_20K)'];
ztest = x_test_20K * B_100;
htest = 1.0./(1.0+exp(-ztest));
%%
for i = 1:20000
    if htest(i) > 0.5
        decision_3_20K(i) = 1;
    else
        decision_3_20K(i) = 0;
    end
end

%% quadratic logistic
N = 10000;
for i = 1:N
    data_10K_qua(i,:) = [x_10K(1,i) x_10K(2,i) (x_10K(1,i))^2 (x_10K(1,i)*x_10K(2,i)) (x_10K(2,i))^2];
end
labels_100_qua = labels_100' +1;
labels_1K_qua = labels_1K' +1;
labels_10K_qua = labels_10K' +1;
%% parameters
B_100_qua = mnrfit(data_100_qua,labels_100_qua);
B_1K_qua = mnrfit(data_1K_qua,labels_1K_qua);
B_10K_qua = mnrfit(data_10K_qua,labels_10K_qua);

%% validation

for i = 1:20000
    data_20K_qua(i,:) = [1 x_20K(1,i) x_20K(2,i) (x_20K(1,i))^2 (x_20K(1,i)*x_20K(2,i)) (x_20K(2,i))^2];
end
ztest_100_qua = data_20K_qua * B_100_qua;
ztest_1K_qua = data_20K_qua * B_1K_qua;
ztest_10K_qua = data_20K_qua * B_10K_qua;
%%
htest = 1.0./(1.0+exp(-ztest_100_qua));

for i = 1:20000
    if htest(i) > 0.5
        decision_3_20K(i) = 0;
    else
        decision_3_20K(i) = 1;
    end
end

%% #of 0 & 1 in sample
N0 = length(find(labels_20K==0)); 
N1 = length(find(labels_20K==1)); 
%% min-Perror classifier
    TP = sum(decision_3_20K==1 & labels_20K==1);
    FN = sum(decision_3_20K==0 & labels_20K==1);
    FP = sum(decision_3_20K==1 & labels_20K==0);
    TN = sum(decision_3_20K==0 & labels_20K==0);
    X_ROC_classifier = FP/(TN+FP);
    Y_ROC_classifier = TP/(TP +FN);
    Pfa = sum(decision_3_20K==1 & labels_20K==0)/N0; 
    Pcd = sum(decision_3_20K==1 & labels_20K==1)/N1; 
    P_error = Pfa*p0+(1-Pcd)*p1; 
    
%% Question 2
 n = 1;
 R = 1;
 x0 = 0;
 y0 = 0;
 sigma_i = 0.3;
 %% true location
 t = 2*pi*rand(n,1);
 r = R*sqrt(rand(n,1));
 xT = x0 + r.*cos(t);
 yT = y0 + r.*sin(t);
 plot(xT,yT,'+')
 hold on;
 h = circle(x0,y0,R);
 
 %% landmarks
 k=3;
 xi = [1; cos(2*pi/3);cos(4*pi/3)];
 yi = [0; sin(2*pi/3);sin(4*pi/3)];
 
 %% range measurments

 for i = 1:k
 dT(i,:) = abs([xT,yT]'-[xi(i,:),yi(i,:)]');
 
 end
 %% equilevel
mu1 = sqrt(dT(1,1)^2+dT(1,2)^2);
mu2 = sqrt(dT(2,1)^2+dT(2,2)^2);
mu3 = sqrt(dT(3,1)^2+dT(3,2)^2);
% mu4 = sqrt(dT(4,1)^2+dT(4,2)^2);
Sigma = 0.3;
x = -3:0.1:3;
y = -3:0.1:3;
[X1,X2] = meshgrid(x,y);
X_1 = sqrt(((X1(:)-xi(1)).^2)+(X2(:)-yi(1)).^2);
X_2 = sqrt(((X1(:)-xi(2)).^2)+(X2(:)-yi(2)).^2);
X_3 = sqrt(((X1(:)-xi(3)).^2)+(X2(:)-yi(3)).^2);
% X_4 = sqrt(((X1(:)-xi(4)).^2)+(X2(:)-yi(4)).^2);
Z1 = normpdf(X_1,mu1,Sigma);
Z2 = normpdf(X_2,mu2,Sigma);
Z3 = normpdf(X_3,mu3,Sigma);
Z4 = normpdf(X_4,mu4,Sigma);
for i = 1:length(Z1)
Z(i) = Z1(i)*Z2(i)*Z3(i);
end
Z = reshape(Z,length(x),length(y));
surf((x),(y),Z)
caxis([min(Z(:))-0.5*range(Z(:)),max(Z(:))])
axis([-2 2 -2 2 0 3])
hold on
plot3(xi(1),yi(1),2,'ro')
hold on
plot3(xi(2),yi(2),2,'ro')
hold on
plot3(xi(3),yi(3),2,'ro')
hold on
% plot3(xi(4),yi(4),2,'ro')
% hold on
plot3(xT,yT,3,'r+')
xlabel('X')
ylabel('Y')
zlabel('likelihood')
title('3 landmarks')
 %%
 function h = circle(x0,y0,R)
hold on
th = 0:pi/50:2*pi;
xunit = R * cos(th) + x0;
yunit = R * sin(th) + y0;
h = plot(xunit, yunit);
 end
%%
function [x,labels] = generateDataA1Q1(N)
%N = 100;
figure(1), clf,     %colors = 'bm'; markers = 'o+';
classPriors = [0.6,0.4];
labels = (rand(1,N) >= classPriors(1));
for l = 0:1
    indl = find(labels==l);
    if l == 0
        N0 = length(indl);
        w0 = [0.5,0.5]; mu0 = [5 0;0 4];
        Sigma0(:,:,1) = [4 0;0 2]; Sigma0(:,:,2) = [1 0;0 3];
        gmmParameters.priors = w0; % priors should be a row vector
        gmmParameters.meanVectors = mu0;
        gmmParameters.covMatrices = Sigma0;
        [x(:,indl),components] = generateDataFromGMM(N0,gmmParameters);
        plot(x(1,indl(components==1)),x(2,indl(components==1)),'mo'), hold on, 
        plot(x(1,indl(components==2)),x(2,indl(components==2)),'go'), hold on, 
        
    elseif l == 1
        m1 = [3;2]; C1 = [2,0;0,2];
        N1 = length(indl);
        x(:,indl) = mvnrnd(m1,C1,N1)';
        plot(x(1,indl),x(2,indl),'b+'), hold on,
        axis equal,
    end
end
end
%%%
function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end