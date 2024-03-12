%%
clc
clear 
close all

load('dataset2.mat');

nk=1; na=2; nb=2;
ts = val.Ts;
uid = id.u;
yid = id.y;
uval = val.u;
yval = val.y;
m=2;

%% PREDICTION MODEL
[modelpred, yid_pred, yval_pred] = predictionARX(id,val,m,na,nb);
for m=1:3 %the system's order won't be higher than 3
    for na=1:7
        %mse error id data - prediction
        mseid_pred = 0;
        for i= 1:length(yid)
            mseid_pred = mseid_pred + (yid_pred(i)-yid(i))^2;
        end
        mseid_pred = 1/length(yid) * mseid_pred;

        %mse error val data - prediction
        mseval_pred = 0;
        for i= 1:length(yval)
            mseval_pred = mseval_pred + (yval_pred(i)-yval(i))^2;
        end
        mseval_pred = 1/length(yval) * mseval_pred;
    end 
    %MSEval_pred(m, na) = mseval_pred;
end
figure
plot(modelpred.y); 
hold on; plot(yval);
legend('model','yval');
title('Prediction model vs y validation');

% figure
% stem(mseid_pred);
% title('MS ERROR FOR VALIDATION DATA - PREDICTION');

%% SIMULATION MODEL
[modelsim, yid_sim, yval_sim]=simulationARX(id,val,m,na,nb);

%mse error identification data- simulation 
% !!! NOT NECESARY
mseid_sim = 0;
for i= 1:length(yid) 
   mseid_sim = mseid_sim + (yid_sim(i)-yid(i))^2;
end
mseid_sim = 1/length(yid) * mseid_sim;

%mse error validation data- simulation
mseval_sim = 0;
for i= 1:length(yval) 
   mseval_sim = mseval_sim + (yval_sim(i)-yval(i))^2;
end
mseval_sim = 1/length(yval) * mseval_sim;

figure
plot(modelsim.y);
hold on, plot(yval);
legend('model','yval');
title('Simulation model vs y validation');

%% FUNCTIONS NEEDED

function [model,yid_sim, yval_sim] = simulationARX(id,val,m,na,nb) %returns the model and the MSE
    ts = val.Ts;    
    uid = id.u;
    yid = id.y;
    uval = val.u;

    phi_predid = reg_generator_pred(uid, yid, m, na, nb);  
    theta = phi_predid \ yid; %theta for prediction identification data

    %simulation model
    yid_sim = reg_generator_sim(uid,m,na,nb,theta);          
    yval_sim = reg_generator_sim(uval,m,na,nb,theta);        

    model = iddata(yval_sim, uval, ts);  
end


function [model,yid_pred,yval_pred] = predictionARX(id, val, m, na, nb) %returns the model and the MSE
    ts = val.Ts;    
    uid = id.u;
    yid = id.y;
    uval = val.u;
    yval = val.y;

    %prediction model
    phi_predid = reg_generator_pred(uid, yid, m, na, nb);  
    theta = phi_predid \ yid;
    
    yid_pred = phi_predid*theta; %yhat prediction id

    phi_predval = reg_generator_pred(uval, yval, m, na, nb);
    yval_pred = phi_predval*theta; %yhat prediction val

    model = iddata(yval_pred, uval, ts);
   
end


function [regressors] = reg_generator_pred(u, y, m, na, nb)
    mat = power_matrix(zeros(1,na+nb), m, []);    
    regressors = zeros(length(u),length(mat));
    for i=1:length(u)
        dk = ones(1,na+nb);
        for a=1:na
            if(i>a)
                dk(a) = -y(i-a);
            else
                dk(a) = 0;
            end
        end
        for b=1:nb
            if(i>b)
                dk(na+b) = u(i-b);
            else
                dk(na+b) = 0;
            end
        end %calculated the delayed phi
        phi_new = ones(1,size(mat,1));
        l = size(mat,1);
        for k = 1:l
            for j = 1:length(dk)
                phi_new(k) = phi_new(k) * dk(j)^mat(k,j);
            end
        end
        regressors(i,:) = phi_new; %all the delayed elements, to the correct powers from the power matrix
    end
end

%regression function for simulation:
function [yhat_sim] = reg_generator_sim(u, m, na, nb, theta)
    yhat_sim = zeros(length(u),1);
    mat = power_matrix(zeros(1,na+nb), m, []);    
    for i=1:length(u)
        dk = zeros(1,na+nb);
        for a=1:na
            if(i>a)
                dk(a) = -yhat_sim(i-a);
            else
                dk(a) = 0;
            end
        end
        for b=1:nb
            if(i>b)
                dk(na+b) = u(i-b);
            else
                dk(na+b) = 0;
            end
        end %calculated the delayed phi (dk)
        %phi must be a row!!!
        phi_new = ones(1,size(mat,1));
        l = size(mat,1);
        for k = 1:l
            for j = 1:length(dk)
                phi_new(k) = phi_new(k) * dk(j)^mat(k,j);
            end
        end
        %all the delayed elements, to the correct powers from the power matrix
        yhat_sim(i) = phi_new*theta;
    end
end


function [matrix] = power_matrix(d, m, matrix) %returns a matrix that generates all the possible power combinations <= to m (recursively)
%if m=0 => returns a vector of 0s
%since they represent the power of the elements in d(k)
%if m>0 we just set the power of one element to m and subtract from it 1 and distribute it to other elements 
%more like a cascade manner ex: 
% [m 0 0 0]->[m-1 1 0 0]->[m-1 0 1 0]->[m-1 0 0 1]->[m-2 2 0 0]->[m-2 1 1 0]->......
    if m == 0
        matrix = zeros(1,length(d));
        return
    else
        matrix = power_matrix(d,m-1,matrix);
        auxd = eye(length(d));
        aux = [];
        for i = 1:length(d)
            for j = 1:size(matrix,1)
                line = matrix(j,:) + auxd(i,:);
                ok = true;
                for k = 1:size(aux,1)
                    if(aux(k,:) == line)
                        ok = false;
                        break;
                    end
                end
                if(ok)
                    aux = [aux; line];
                end
            end
        end
        for i = 1:size(matrix,1)
            j= 1;
            while j <= size(aux,1)
                if(matrix(i,:) == aux(j,:))
                    aux(j,:) = [];
                end
                j = j+1;
            end
        end
        matrix = [matrix; aux];
    
    end
end
