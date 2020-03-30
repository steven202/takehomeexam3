clear all; close all; clc;
rng('default');
rng(1);

c=3;
n_train=100;
n_test=1000;
k=10;

[x_train,y_train] = generateMultiringDataset(c,n_train);
[x_test, y_test] = generateMultiringDataset(c, n_test);
iter_max = 100;
result = zeros(iter_max,3);
x_train_label1=x_train(:,y_train==1);
x_train_label2=x_train(:,y_train==2);
x_train_label3=x_train(:,y_train==3);
priors_train=[size(x_train_label1,2)/n_train size(x_train_label2,2)/n_train size(x_train_label3,2)/n_train];
result = goloop(x_train_label1,result,1,iter_max,k);
result = goloop(x_train_label2,result,2,iter_max,k);
result = goloop(x_train_label3,result,3,iter_max,k);
%%
gmm_ans = mode(result,1);
disp(gmm_ans);

[x1_alpha,x1_mu,x1_sigma]=init_params(7,x_train_label1);
[logLikelihood1,alpha1,mu1,Sigma1]=EMforGMM(x_train_label1,x1_alpha,x1_mu,x1_sigma,x_train_label1);

[x2_alpha,x2_mu,x2_sigma]=init_params(7,x_train_label2);
[logLikelihood2,alpha2,mu2,Sigma2]=EMforGMM(x_train_label2,x2_alpha,x2_mu,x2_sigma,x_train_label2);

[x3_alpha,x3_mu,x3_sigma]=init_params(10,x_train_label3);
[logLikelihood3,alpha3,mu3,Sigma3]=EMforGMM(x_train_label3,x3_alpha,x3_mu,x3_sigma,x_train_label3);
%%
[x_alpha_t,x_mu_t,x_sigma_t]=init_params(7+7+10,x_train);
[logLikelihood_t,alpha_t,mu_t,Sigma_t]=EMforGMM(x_train,x_alpha_t,x_mu_t,x_sigma_t,x_train);

% MU=[x1_mu, x2_mu,x3_mu];
% SIGMA=cat(3, x1_sigma,x2_sigma,x3_sigma);
% gm1=gmdistribution(MU',SIGMA);
gm1=gmdistribution(x_mu_t',x_sigma_t);
p1 = posterior(gm1, x_train');
[max_post1,loglikelihood1]=max(p1');
train_out=zeros(1,size(x_train,2));
for i=1:size(x_train,2)
    if loglikelihood1(1,i)<=7
        train_out(1,i)=1;
    elseif  loglikelihood1(1,i)<=7+7
        train_out(1,i)=2;
    else
        train_out(1,i)=3;
    end
end
Y=y_train;
H=train_out;
N=n_train;
loss = crossentropy(H,Y);
disp(loss);

%%
get 
gm2=gm1;
p2 = posterior(gm2, x_test');
[max_post2,loglikelihood2]=max(p2');
test_out=zeros(1,size(x_test,2));
for i=1:size(x_test,2)
    if loglikelihood2(1,i)<=7
        test_out(1,i)=1;
    elseif  loglikelihood2(1,i)<=7+7
        test_out(1,i)=2;
    else
        test_out(1,i)=3;
    end
end
Y2=y_test;
H2=test_out;
N2=n_test;
loss2 = crossentropy(H2,Y2);
disp(loss2);

%%

function result = goloop(x_train, result, num,iter_max,k)
    n_train=size(x_train,2);
    for i=1:iter_max
        pp = n_train/k;
        for k_fold=1:k
            x_validate_tmp = x_train(:, (k_fold-1)*pp+1:k_fold*pp);
            x_train_tmp = [x_train(:, 1:(k_fold-1)*pp) x_train(:, k_fold*pp+1:n_train)];
            [winner]=gmm_run(x_train_tmp, x_validate_tmp);
            temp(k_fold)=winner;
        end
        result(i,num)=max(temp);
    end
end

% for i=1:iter_max
%     pp = n_train/k;
%     for k_fold=1:k
%         x_validate_tmp = x_train(:, (k_fold-1)*pp+1:k_fold*pp);
%         y_validate_tmp = y_train(:, (k_fold-1)*pp+1:k_fold*pp);
%         x_train_tmp = [x_train(:, 1:(k_fold-1)*pp) x_train(:, k_fold*pp+1:n_train)];
%         y_train_tmp = [y_train(:, 1:(k_fold-1)*pp) y_train(:, k_fold*pp+1:n_train)];
%         [winner]=gmm_run(x_train_tmp, x_validate_tmp);
%         temp(k_fold)=winner;
%     end
%     result(i,2)=max(temp);
% end

%%
%x3 = randGMM(1000,alpha_true,mu_true,Sigma_true);
% x1_train(1,:) = datasample(s,x1(1,:),n1,'Replace',true);


function winner1=gmm_run(x_train, x_vali)
    %s = RandStream('mlfg6331_64');   
    for M=1:15
        [x1_alpha,x1_mu,x1_sigma]=init_params(M,x_train);
        [logLikelihood1,alpha1,mu1,Sigma1]=EMforGMM(x_train,x1_alpha,x1_mu,x1_sigma,x_vali);
        logLikelihoods(M)=logLikelihood1;
    end
    winner1=find(logLikelihoods==max(logLikelihoods),1);
end
%%
function [alpha,mu,Sigma]=init_params(M,x)
    delta = 1e-2; % tolerance for EM stopping criterion
    regWeight = 1e-10; % regularization parameter for covariance estimates
    N=size(x,2);
    % Initialize the GMM to randomly selected samples
    alpha = ones(1,M)/M;
    shuffledIndices = randperm(N);
    mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
    for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(2,2);
    end
end
%%
function [logLikelihood,alpha,mu,Sigma]=EMforGMM(x,alpha,mu,Sigma,x_validate)
% Generates N samples from a specified GMM,
% then uses EM algorithm to estimate the parameters
% of a GMM that has the same nu,mber of components
% as the true GMM that generates the samples.

close all,
delta = 1e-2; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
%%%%%%%%%%%%%%%%%%%%%% generate data %%%%%%%%%%%%%%%%%
% % Generate samples from a 3-component GMM
% mu_true = [-10 0 10 1;0 2 0 -3];
% 
% 
% [d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components

t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 1; % Not converged at the beginning
while ~Converged && t<=1000
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
end
logLikelihood = sum(log(evalGMM(x_validate,alpha,mu,Sigma)));
end

%%
function x = randGMM(N,alpha,mu,Sigma)
    d = size(mu,1); % dimensionality of samples
    cum_alpha = [0,cumsum(alpha)];
    u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
    for m = 1:length(alpha)
        ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
        x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    end
end

%%
function x = randGaussian(N,mu,Sigma)
    % Generates N samples from a Gaussian pdf with mean mu covariance Sigma
    n = length(mu);
    z =  randn(n,N);
    A = Sigma^(1/2);
    x = A*z + repmat(mu,1,N);
end

%%
function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
    x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
    x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
    [h,v] = meshgrid(x1Grid,x2Grid);
    GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
    zGMM = reshape(GMM,91,101);
    %figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end


%%
function gmm = evalGMM(x,alpha,mu,Sigma)
    gmm = zeros(1,size(x,2));
    for m = 1:length(alpha) % evaluate the GMM on the grid
        gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
    end
end

%%
function g = evalGaussian(x,mu,Sigma)
    % Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
    [n,N] = size(x);
    Sigma(isnan(Sigma))=-1000;
    invSigma = pinv(Sigma);
    C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
end
