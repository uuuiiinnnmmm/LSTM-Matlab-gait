% CODE 1 : preprocess.m
% -------------------------------------------------------------------------
% 描述:
%   1. 从多个.mat文件直接加载原始IMU步态数据。
%   2. 对数据进行分窗、标注、归一化和数据集划分。
%   3. 将预处理完成的数据保存为 "preprocessed_data.mat"。
% -------------------------------------------------------------------------

clear; clc; close all;

%% =================== Part 1: 加载 .mat 原始数据 ====================
disp('Part 1: Loading raw .mat files...');

rawDataFolder = 'RawData'; 

% 获取所有.mat文件
matFiles = dir(fullfile(rawDataFolder, '*.mat'));

% 检查是否找到了文件
if isempty(matFiles)
    error('No .mat files found in the "%s" folder. Please check the path and file locations.', rawDataFolder);
end

rawData = cell(1, length(matFiles));
rawLabels = cell(1, length(matFiles));

for i = 1:length(matFiles)
    fileName = matFiles(i).name;
    filePath = fullfile(rawDataFolder, fileName);
    
    % 加载.mat文件
    loadedData = load(filePath);
    
    if isfield(loadedData, 'data')
        imuData = loadedData.data;
    else
        % 如果找不到名为'data'的变量，程序会报错并提示
        error('Variable "data" not found in file: %s. Please check the variable name inside your .mat file.', fileName);
    end
    
    % 从文件名提取标签
    [~, labelName, ~] = fileparts(fileName);
    
    % 将数据和标签存储到cell数组中
    rawData{i} = imuData;
    rawLabels{i} = labelName;
    
    fprintf('Successfully loaded data from "%s".\n', fileName);
end
disp('All .mat files loaded.');
disp('---------------------------------');

%% =================== Part 2: 数据预处理 ====================
disp('Part 2: Starting data preprocessing...');

% --- 用户配置 ---
windowSize = 128;       % 每个步态序列的长度
overlapPercentage = 0.5;% 窗口重叠率
overlapLength = floor(windowSize * overlapPercentage);

% --- 分窗与标注 ---
segments = {};
segmentLabels = [];

for i = 1:length(rawData)
    data = rawData{i};
    label = rawLabels{i};
    
    idx = 1;
    while (idx + windowSize - 1) <= size(data, 1)
        segment = data(idx : idx + windowSize - 1, :);
        % MATLAB的LSTM层需要特征在行，时间步在列，所以转置
        segments{end+1} = segment'; 
        segmentLabels{end+1} = label;
        idx = idx + (windowSize - overlapLength);
    end
end
fprintf('Data windowing complete. Generated %d segments.\n', length(segments));

% 将标签转为 categorical 类型，这是分类任务的标准做法
segmentLabels = categorical(segmentLabels');

% --- 数据集划分 (70% 训练, 30% 测试) ---
cv = cvpartition(segmentLabels, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

trainData = segments(trainIdx);
trainLabels = segmentLabels(trainIdx);
testData = segments(testIdx);
testLabels = segmentLabels(testIdx);

% --- 数据归一化 ---
% 只使用训练数据计算均值和标准差
fprintf('Normalizing data based on training set statistics...\n');
allTrainData = cat(3, trainData{:});
mu = mean(allTrainData, [2 3]);
sigma = std(allTrainData, 0, [2 3]);

% 应用归一化到训练集和测试集
for i = 1:length(trainData)
    trainData{i} = (trainData{i} - mu) ./ sigma;
end

for i = 1:length(testData)
    testData{i} = (testData{i} - mu) ./ sigma;
end
disp('Normalization complete.');

%% ====================== Part 3: 保存预处理数据 =======================
outputProcessedFile = 'preprocessed_gait_data.mat';
save(outputProcessedFile, 'trainData', 'trainLabels', 'testData', 'testLabels', 'mu', 'sigma');

fprintf('Preprocessing finished.\n');
fprintf('Processed data has been saved to "%s".\n', outputProcessedFile);
disp('You can now run "train_gait_lstm.m" to build and train the model.');
