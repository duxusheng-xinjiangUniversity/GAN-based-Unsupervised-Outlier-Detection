function [outputArg1,outputArg2] = MyBasicGAN(train_x, iteration)

[m,n]= size(train_x);
%-----------Defining the model
generator=nnsetup([n,2,n]);
discriminator=nnsetup([n,6,1]);
%----------Parameter Setting
batch_size= m; 
images_num= m;
batch_num= floor(images_num / batch_size);
learning_rate= 0.001;

    for i = 1: iteration
        kk = randperm(images_num);% Generate m random numbers without repetition, disrupting the entire sequence of the sample
    %Prepare data
        images_real = train_x;
        noise = unifrnd(0,1,m,n);
        %Start training
        generator = nnff(generator, noise); 
        images_fake = generator.layers{generator.layers_count}.a; 
        discriminator = nnff(discriminator,images_fake);
        logits_fake = discriminator.layers{discriminator.layers_count}.z; 
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));
        generator = nnbp_g(generator, discriminator);
        generator = nnapplygrade(generator, learning_rate);
        generator = nnff(generator, noise);
        images_fake = generator.layers{generator.layers_count}.a;
        images = [images_fake; images_real];
        discriminator = nnff(discriminator, images); 
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(batch_size,1); ones(batch_size,1)];
        discriminator = nnbp_d(discriminator, logits, labels);
        discriminator = nnapplygrade(discriminator, learning_rate);
        c_loss(i,:) = sigmoid_cross_entropy(logits(1:batch_size), ones(batch_size,1));
        d_loss (i,:)= sigmoid_cross_entropy(logits, labels);
        if corr2(images_fake,images_real)>0.6
            break;
        end
        %%%%%%End Train%%%%%
        if i==iteration
            wm = sprintf('fake_data.txt');
            filename = ['D:\matlab2019a\matlab files\GAN for Outlier Detection\GAN+Other for Outlier Detection 0510\',wm];
            dlmwrite(filename,images_fake,'delimiter',' ');
        end
     end
end



% sigmoid
function output = sigmoid(x)
    output = 1 ./(1+exp(-x));
end

%relu
function output = relu(x)
    output = max(x,0);
end


function output = delta_relu(x)
    output = max(x,0);
    output(output>0) = 1;
end


function result = sigmoid_cross_entropy(logits,labels)
    result = max(logits,0) - logits .* labels + log(1+exp(-abs(logits)));
    result = mean(result);
end


function result = delta_sigmoid_cross_entropy(logits,labels)
    temp1 = max(logits, 0);
    temp1(temp1>0) = 1;
    temp2 = logits;
    temp2(temp2>0) = -1;
    temp2(temp2<0) = 1;
    result = temp1- labels +exp(-abs(logits))./(1+exp(-abs(logits))) .*temp2;
end


function nn = nnsetup(architecture)
    nn.architecture = architecture;
    nn.layers_count = numel(nn.architecture);
    %adam
    nn.t=0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    %----------------------
    for i=2 : nn.layers_count
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i));
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end

%Forward Propagation
function nn = nnff(nn, x)
    nn.layers{1}.a = x;
    for i = 2 : nn.layers_count
        input = nn.layers{i-1}.a;
        w = nn.layers{i}.w;
        b = nn.layers{i}.b;
        nn.layers{i}.z = input *w + repmat(b, size(input,1), 1); 
        if i ~= nn.layers_count
            nn.layers{i}.a = relu(nn.layers{i}.z); 
        else
            nn.layers{i}.a = sigmoid(nn.layers{i}.z);
        end
    end

end


function nn = nnbp_d(nn, y_h, y)

    n = nn.layers_count;

    nn.layers{n}.d = delta_sigmoid_cross_entropy(y_h, y); 
    for i = n-1 : -1:2 
        d = nn.layers{i+1}.d;
        w = nn.layers{i+1}.w;
        z = nn.layers{i}.z;
       
        nn.layers{i}.d = d * w' .* delta_relu(z);
    end
    

    for i = 2: n
        d = nn.layers{i}.d;
        a = nn.layers{i-1}.a;

        nn.layers{i}.dw = a'*d /size(d,1);
        nn.layers{i}.db = mean(d,1);
    end
end


function g_net = nnbp_g(g_net, d_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;

    g_net.layers{n}.d = d_net.layers{2}.d * d_net.layers{2}.w' .* (a .* (1-a));
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;

        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end

    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;

        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);
    end
end




%Adam
function nn = nnapplygrade(nn, learning_rate);
    n = nn.layers_count;
    nn.t = nn.t+1;
    beta1 = nn.beta1;
    beta2 = nn.beta2;
    lr = learning_rate * sqrt(1-nn.beta2^nn.t) / (1-nn.beta1^nn.t);
    for i = 2:n
        dw = nn.layers{i}.dw;
        db = nn.layers{i}.db;

        nn.layers{i}.w_m = beta1 * nn.layers{i}.w_m + (1-beta1) * dw;
        nn.layers{i}.w_v = beta2 * nn.layers{i}.w_v + (1-beta2) * (dw.*dw);
        nn.layers{i}.w = nn.layers{i}.w -lr * nn.layers{i}.w_m ./ (sqrt(nn.layers{i}.w_v) + nn.epsilon);
        nn.layers{i}.b_m = beta1 * nn.layers{i}.b_m + (1-beta1) * db;
        nn.layers{i}.b_v = beta2 * nn.layers{i}.b_v + (1-beta2) * (db .* db);
        nn.layers{i}.b = nn.layers{i}.b -lr * nn.layers{i}.b_m ./ (sqrt(nn.layers{i}.b_v) + nn.epsilon);        
    end
    
end







%%initialization
function parameter=initializeGaussian(parameterSize,sigma)
if nargin < 2
    sigma=0.05;
end
parameter=randn(parameterSize,'double') .* sigma;
end


