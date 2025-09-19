% 读取Excel文件
filename = '步态数据集.xlsx';
data = readtable(filename);

% 数据清洗
data = rmmissing(data); % 去除缺失值

% 分离特征和标签
X = data(:, 1:9);
Y = data(:, 10);

% 将表格数据转换为数组
X = table2array(X);
Y = categorical(table2array(Y));

% 数据预处理：中值滤波
windowSize = 5;
X = medfilt1(X, windowSize);

% 数据标准化：Z-score标准化
X = zscore(X);

% 划分训练集和测试集
[trainInd,valInd,testInd] = dividerand(size(X, 1), 0.8, 0.1, 0.1);
X_train = X(trainInd, :);
Y_train = Y(trainInd, :);
X_val = X(valInd, :);
Y_val = X(valInd,:);
X_test = X(testInd, :);
Y_test = Y(testInd, :);

% 定义深度学习网络结构
layers = [
    featureInputLayer(9) 
    lstmLayer(150, 'OutputMode', 'last') 
    fullyConnectedLayer(250) % 神经元数量
    batchNormalizationLayer 
    reluLayer
    dropoutLayer(0.3) % 调整dropout率
    fullyConnectedLayer(250)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numel(categories(Y)))
    softmaxLayer
    classificationLayer];

% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 180, ... 
    'MiniBatchSize', 128, ... 
    'ValidationData', {X_test, Y_test}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'InitialLearnRate', 0.002, ... 
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ... % 调整学习率下降因子
    'LearnRateDropPeriod', 70, ... % 调整学习率下降周期
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'auto');


% 训练最终网络
net = trainNetwork(X_train, Y_train, layers, options);

% 测试网络
Y_pred = classify(net, X_test);
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
disp(['测试集准确率: ', num2str(accuracy)]);

% 保存模型
save('步态识别模型.mat', 'net');

% 显示混淆矩阵
figure;
confusionchart(Y_test, Y_pred);
title('混淆矩阵');

% 极值标准化
function out=mystand(A)
out=[];
n=size(A,1);%获取行数
minA = min(A); %获取极小值
maxA = max(A);%获取极大值
out = (A-repmat(minA,n,1))./repmat(maxA-minA,n,1);%使用repmat对每个元素进行重复处理
end

% z标准化
function out=myzscore(A)
temp =[];
demesion=size(A);
meanA=mean(A);
stdA=std(A);
for i=1:numel(A)
temp = [temp (A(i)-meanA)/stdA];
end
out=reshape(temp,demesion);
end
