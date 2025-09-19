% SCRIPT 1: preprocess_gait_data.m
% -------------------------------------------------------------------------
% 描述:
%   1. 从多个.mat文件加载原始IMU步态数据。
%   2. 将每个.mat文件的数据保存为一个单独的Excel工作表。
%   3. 从生成的Excel文件中读取数据进行后续处理。
%   4. 对数据进行分窗、标注、归一化和数据集划分。
%   5. 将预处理完成的数据保存为 "preprocessed_data.mat"。
% -------------------------------------------------------------------------

clear; clc; close all;

%% =================== Part 1: 加载.mat并转换为Excel ====================
disp('Part 1: Loading .mat files and converting to Excel...');

% --- 用户配置 ---
rawDataFolder = 'RawData'; 
outputExcelFile = 'GaitData.xlsx';

% 获取所有.mat文件
matFiles = dir(fullfile(rawDataFolder, '*.mat'));

% 检查当前目录下是否已存在同名Excel文件，如果存在则删除
if exist(outputExcelFile, 'file')
    delete(outputExcelFile);
    fprintf('Existing Excel file "%s" has been deleted.\n', outputExcelFile);
end

rawData = cell(1, length(matFiles));
rawLabels = cell(1, length(matFiles));

for i = 1:length(matFiles)
    fileName = matFiles(i).name;
    filePath = fullfile(rawDataFolder, fileName);
    
    % 加载.mat文件
    loadedData = load(filePath);
    
    imuData = loadedData.data; 
    
    % 从文件名提取标签 (例如 'UserA.mat' -> 'UserA')
    [~, labelName, ~] = fileparts(fileName);
    
    % 将数据写入Excel的不同工作表(Sheet)
    writetable(array2table(imuData), outputExcelFile, 'Sheet', labelName);
    
    % 将数据和标签存储到cell数组中以备后用
    rawData{i} = imuData;
    rawLabels{i} = labelName;
    
    fprintf('Loaded "%s" and saved data to sheet "%s" in "%s"\n', fileName, labelName, outputExcelFile);
end
disp('Excel file conversion complete.');
disp('---------------------------------');


%% =================== Part 2: 读取数据并进行预处理 ====================
% 在这一部分，我们直接使用上一步加载到内存中的`rawData`和`rawLabels`

% % --- 从Excel文件读取数据 ---
% disp('Part 2: Reading data from Excel for preprocessing...');
% sheets = sheetnames(outputExcelFile);
% rawDataFromExcel = cell(1, length(sheets));
% rawLabelsFromExcel = cell(1, length(sheets));
% for i = 1:length(sheets)
%     rawDataFromExcel{i} = readmatrix(outputExcelFile, 'Sheet', sheets{i});
%     rawLabelsFromExcel{i} = sheets{i};
% end
% % 使用从Excel读取的数据
% rawData = rawDataFromExcel; 
% rawLabels = rawLabelsFromExcel;

disp('Part 2: Starting data preprocessing...');

% --- 用户配置 ---
windowSize = 128;       % 每个步态序列的长度 (时间步)
overlapPercentage = 0.5;% 窗口重叠率 (50%)
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
        % MATLAB的LSTM层需要特征在行，时间步在列
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
