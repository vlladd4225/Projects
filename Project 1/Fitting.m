clc
clear
close all
load ("dataset1.mat");

m_max = 12; %maxim degree of the polynomial

% 1.Identification data
xid = id.X{1};
xid2 = id.X{2};
yid = id.Y;

l1=length(xid);
%id_mse_vector = zeros(1,m_max);
for m=1:m_max
    phi_id = zeros(l1*l1, (m+1)*(m+2)/2);
    phi_col = 1;
    phi_row = 1;
    for vect1_index = 1:l1
        x1 = xid(vect1_index);
        for vect2_index = 1:l1
            x2 = xid2(vect2_index);
            %generate phi elements
            phi_col = 1;
            for i=0:m
                for j=0:i 
                    prod = x1^(i-j);
                    phi_id(phi_row,phi_col) = prod * x2^j;
                    phi_col = phi_col+1;
                end
            end 
            phi_row = phi_row+1;
        end
    end

    theta_id = phi_id\yid(:);
    yhat_id = phi_id * theta_id;
    
    %transform yid from a matrix to a vector
    vector_yid = reshape(yid, 1, []);

    %reshape y_hat into a matrix
    matrix_yhat = reshape(yhat_id, l1, l1);
    
    %calculate mean squared error
    mse=0;
    for i= 1:length(vector_yid) 
        mse = mse + (abs(yhat_id(i)-vector_yid(i)))^2;
    end
    mse = 1/length(vector_yid)*mse;
    id_mse_vector(m)= mse;
end
MSE_min_id = min(id_mse_vector);

for i = 1:length(id_mse_vector)
   if id_mse_vector(i) == MSE_min_id
       ideal_degree_id = i;
   end
end

%plot identification data to compare with the approximated values
figure;
subplot(1,2,1);
mesh(xid, xid2, yid, 'FaceColor','r');
title('Identification data');
subplot(1,2,2);
mesh(xid, xid2, matrix_yhat, 'FaceColor','b');
title('Yhat Approximation')

% 2. Validation data
val_x1 = val.X{1};
val_x2 = val.X{2};
y_validation = val.Y;

l2=length(val_x1);
%val_mse_vector = zeros(1,m_max);

for m=1:m_max
    phi_val = zeros(l2*l2, (m+1)*(m+2)/2);
    phi_col = 1;
    phi_row = 1;
    for vect1_index = 1:l2
        x1 = val_x1(vect1_index);
        for vect2_index = 1:l2
            x2 = val_x2(vect2_index);
            %generate phi elements
            phi_col = 1;
            for i=0:m
                for j=0:i 
                    prod = x1^(i-j);
                    phi_val(phi_row,phi_col) = prod * x2^j;
                    phi_col = phi_col+1;
                end
            end 
            phi_row = phi_row+1;
        end
    end

    theta_val = phi_val\y_validation(:);
    yhat_val = phi_val * theta_val;

    %transform yval from a matrix to a vector
    vector_yval = reshape(y_validation, 1, []);

    %reshape y_hat into a matrix
    matrix_valyhat = reshape(yhat_val, l2, l2);
    
    %calculate mse
    mse_val=0;
    for i= 1:length(vector_yval) 
        mse_val = mse_val + (abs(yhat_val(i)-vector_yval(i)))^2;
    end
    mse_val = 1/length(vector_yval)*mse_val;
    val_mse_vector(m)= mse_val;
end

MSE_min_val = min(val_mse_vector);

for i = 1:length(val_mse_vector)
   if val_mse_vector(i) == MSE_min_val
       ideal_degree_val = i;
   end
end

%plot validation data vs approximated values
figure;
subplot(1,2,1);
mesh(val_x1, val_x2, y_validation, 'FaceColor','g');
title('Validation data')
subplot(1,2,2);
mesh(val_x1, val_x2, matrix_valyhat, 'FaceColor','c');
title('Yhat Approximation')

%final of plot mse depending on the degree m
figure;
plot(1:m_max, id_mse_vector);
hold;
plot(1:m_max, val_mse_vector);
title('MSE plot depending on grade m');
legend('MSE id','MSE val')

%% ideal degree
ideal_m = ideal_degree_val;

phi_id = zeros(l1*l1, (ideal_m+1)*(ideal_m+2)/2);
phi_col = 1;
phi_row = 1;
for vect1_index = 1:l1
    x1 = xid(vect1_index);
    for vect2_index = 1:l1
        x2 = xid2(vect2_index);
        %generate phi elements
        phi_col = 1;
        for i=0:m
            for j=0:i 
                prod = x1^(i-j);
                phi_id(phi_row,phi_col) = prod * x2^j;
                phi_col = phi_col+1;
             end
        end 
        phi_row = phi_row+1;
    end
end

theta_id = phi_id\yid(:);
yhat_id = phi_id * theta_id;

%reshape y_hat into a matrix
matrix_yhat = reshape(yhat_id, l1, l1);

figure
mesh(xid, xid2, yid, 'FaceColor','g');
hold
mesh(xid, xid2, matrix_yhat, 'FaceColor','b');
title('Identification vs Yhat Approximation for min MSE');
figure


phi_val = zeros(l2*l2, (ideal_m+1)*(ideal_m+2)/2);
phi_col = 1;
phi_row = 1;

for vect1_index = 1:l2
    x1 = val_x1(vect1_index);
    for vect2_index = 1:l2
        x2 = val_x2(vect2_index);
         %generate phi elements
         phi_col = 1;
         for i=0:ideal_m
             for j=0:i 
                  prod = x1^(i-j);
                  phi_val(phi_row,phi_col) = prod * x2^j;
                  phi_col = phi_col+1;
              end
         end 
         phi_row = phi_row+1;
     end
end
theta_val = phi_val\y_validation(:);
yhat_val = phi_val * theta_val;

%reshape y_hat into a matrix
matrix_valyhat = reshape(yhat_val, l2, l2);

mesh(val_x1, val_x2, y_validation, 'FaceColor','m');
hold
mesh(val_x1, val_x2, matrix_valyhat, 'FaceColor','b');
title('Validation vs Yhat Approximation for min MSE');