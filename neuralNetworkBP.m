function [] = neuralNetworkBP (act_type,numberOfHiddenLevel,numberOfHiddenNodes,iterations,inputNodes,outputNodes,batchSize,learningRate,momentum)
% to load iris data
load irisData.dat;
sampleSize = length(irisData);

%normalize dataset from value 0.9 to 0.1
a = min(irisData(:,1:4));
b = max(irisData(:,1:4));
ra = 1;
rb = 0;
for rowId = 1:length(irisData)
    irisData(rowId,1:4) = (((ra-rb) * (irisData(rowId,1:4) - a)) / (b - a)) + rb;
end

%changed it to static set of training and test error
trainingSet = cat(1,irisData(1:40,:),irisData(51:90,:));
trainingSet = cat(1,trainingSet,irisData(101:140,:));

testSet = cat(1,irisData(41:50,:),irisData(91:100,:));
testSet = cat(1,testSet,irisData(141:150,:));

%Creating Output matrix Training set
tempOutPutSet = trainingSet(:,5);
outPutSet = ones(length(tempOutPutSet),3);
for rowid = 1:length(tempOutPutSet)
    if tempOutPutSet(rowid,1) == 0
        x = [0 0 1];
    elseif  tempOutPutSet(rowid,1) == 10
        x = [0 1 0];
    else   
         x = [1 0 0];    
    end
    outPutSet(rowid,:) = x;
end


%Creating Output matrix - Test Set
tempOutPutSetTS = testSet(:,5);
outPutSetTS = ones(length(tempOutPutSetTS),3);
for rowid = 1:length(tempOutPutSetTS)
    if tempOutPutSetTS(rowid,1) == 0
        x = [0 0 1];
    elseif  tempOutPutSetTS(rowid,1) == 10
        x = [0 1 0];
    else   
         x = [1 0 0];    
    end
    outPutSetTS(rowid,:) = x;
end

%Creating Training Set Batch
numOfBatch = length(trainingSet)/batchSize;
for batchNum = 1:numOfBatch
     batch{batchNum} = trainingSet(((batchNum*batchSize)-batchSize)+1:batchNum*batchSize,:);  
     outputBatch{batchNum} = outPutSet(((batchNum*batchSize)-batchSize)+1:batchNum*batchSize,:);
end

%variable declaration
M = inputNodes;
N = numberOfHiddenNodes;
O = outputNodes;

%Generating weight for Level 1
WtMatrix{1} = rand(M+1,N(1,1));
%Generating weight for all Hidden Layers
for k = 1:numberOfHiddenLevel-1
    WtMatrix{k+1} = rand(N(1,k)+1,N(1,k+1));
end
%Generating weight for output level
WtMatrix{numberOfHiddenLevel+1} = rand(N(1,numberOfHiddenLevel)+1,O);

for k = 1:numberOfHiddenLevel+1
    WtMatrixOfPrevItr{k} = [];
end

% Creating Activation Function
switch upper(act_type)
    case 'SIGMOID'
        actf = @(x) 1/(1+exp(-x));
        deri = @(y) y.*(1-y);
        
    case 'TANH'
         actf = @(x) tanh(x);
         deri = @(y) 1- y.^2;
    case 'RELU'
       actf = @(x) log(1+exp(x)); 
       deri = @(y) (exp(y)-1)./exp(y);
end


% Matrices for plot
for hiddenLayer = 1 : numberOfHiddenLevel
    actStorage{hiddenLayer} = [];
end

for hiddenLayer = 1 : numberOfHiddenLevel
    wtStorage{hiddenLayer} = [];
end

errorSet4Plot = ones(length(iterations),3);

for hiddenLy = 1 : numberOfHiddenLevel
    wtChSet4Plot{hiddenLy} = ones(length(iterations),3);
end

%Training
    for itr = 1:iterations %For User defined rotation%
        for batchLevel = 1:numOfBatch
            input{1}  = batch{batchLevel}(:,1:4);
            %feedforwad
            for level = 1:numberOfHiddenLevel+1 %For number of levels%  
                input{level} = [input{level} ones(length(input{level}),1)]; % add bias to input
                dotProduct{level} = input{level}*WtMatrix{level}; % input and wt matrix multiplication
                output{level} = arrayfun(actf,dotProduct{level}); % apply activation function
                input{level+1} = output{level};
                errDerivative{level} = arrayfun(deri,output{level});% elementwise derivative
            end 
            %backpropagation
             for level = (numberOfHiddenLevel+1):-1:1
                    if level == (numberOfHiddenLevel+1)
                         exactError = outputBatch{batchLevel} - output{numberOfHiddenLevel+1}; % elementwise(tk – Ok)
                         errorAtEachOutputNode{level} = times(errDerivative{level},exactError); % derivative *(tk – Ok)
                    else 
                        tempWt = transpose(WtMatrix{level+1});
                        avgErrorAtPrevLayertemp = errorAtEachOutputNode{level+1}*tempWt(:,1:(size(tempWt,2)-1));% Wkh ?k
                        errorAtEachOutputNode{level} = times(errDerivative{level},avgErrorAtPrevLayertemp);% derivative * ?k  Wkh ?k
                    end
             end
             
             % adjustWt
             for level = 1:(numberOfHiddenLevel+1)
                 if isempty(WtMatrixOfPrevItr{level}) 
                    deltaW{level} = learningRate *(transpose(input{level})* errorAtEachOutputNode{level});
                 else
                     deltaW{level} = learningRate *(transpose(input{level})* errorAtEachOutputNode{level}) +(momentum*WtMatrixOfPrevItr{level});
                 end    
                 WtMatrixOfPrevItr{level} = deltaW{level};
                 WtMatrix{level}= WtMatrix{level} + deltaW{level} ;
             end
        end
        
        %Classification error for training and test at the end of each iteration
        %feedforward training set
        inputForTrainingTest{1} = trainingSet(:,1:4);
            for level = 1:numberOfHiddenLevel+1 %For number of levels%  
                inputForTrainingTest{level} = [inputForTrainingTest{level} ones(length(inputForTrainingTest{level}),1)]; % add bias to input
                dotProductForTrainingTest{level} = inputForTrainingTest{level}*WtMatrix{level}; % input and wt matrix multiplication
                outputForTrainingTest{level} = arrayfun(actf,dotProductForTrainingTest{level}); % apply activation function
                inputForTrainingTest{level+1} = outputForTrainingTest{level};
                if level ~= numberOfHiddenLevel+1
                    actStorage{level} = cat(1,actStorage{level},outputForTrainingTest{level});
                end
            end 
            outputForTrainingTestFinal = outputForTrainingTest{numberOfHiddenLevel+1};
            error_TRS = outPutSet - outputForTrainingTestFinal;
            sseTRS = trace(error_TRS'*error_TRS)/size(outputForTrainingTestFinal,1); % MSEof training set
            
           %feedforward testing test
           inputForTestingSetTest{1} = testSet(:,1:4);
            for level = 1:numberOfHiddenLevel+1 %For number of levels%  
                inputForTestingSetTest{level} = [inputForTestingSetTest{level} ones(length(inputForTestingSetTest{level}),1)]; % add bias to input
                dotProductForTestingSetTest{level} = inputForTestingSetTest{level}*WtMatrix{level}; % input and wt matrix multiplication
                outputForTestingSetTest{level} = arrayfun(actf,dotProductForTestingSetTest{level}); % apply activation function
                inputForTestingSetTest{level+1} = outputForTestingSetTest{level};
                
            end 
            outputForTestingSetFinal = outputForTestingSetTest{numberOfHiddenLevel+1};  
            error_TS = outPutSetTS - outputForTestingSetFinal;
            sseTS = trace(error_TS'*error_TS)/size(outputForTestingSetFinal,1);  % MSE of test set
           
            %Collecting error over iteration
            calError = [itr sseTRS  sseTS];
            errorSet4Plot(itr,:) = calError;
            
           %weight change angle calculation
            if itr ~= 1
                for lvl = 1:numberOfHiddenLevel
                    wtchangeAtNode1 = acos(dot(WtMatrix{lvl}(:,1),wtStorage{lvl}(:,1))/norm((WtMatrix{lvl}(:,3)))*norm(wtStorage{lvl}(:,3)));
                    wtchangeAtNode2 = acos(dot(WtMatrix{lvl}(:,2),wtStorage{lvl}(:,2))/norm((WtMatrix{lvl}(:,2)))*norm(wtStorage{lvl}(:,2)));
                    tempp = [itr wtchangeAtNode1 wtchangeAtNode2];
                    wtChSet4Plot{lvl}(itr,:) = tempp;
                end 
            end    
            for hiddenLayer = 1 : numberOfHiddenLevel
                wtStorage{hiddenLayer} = WtMatrix{hiddenLayer};
            end
    end
    % plot Training and Testing error
    plotyy(errorSet4Plot(:,1),errorSet4Plot(:,2),errorSet4Plot(:,1),errorSet4Plot(:,3));
    
    % Histogram for two hidden unit from each layer
    for lvl = 1:numberOfHiddenLevel
        for node = 1:2
            hist(actStorage{lvl}(:,node));
        end
    end
    
    % Wt. change plot for two hidden unit from each layer
     for lvl = 1:numberOfHiddenLevel
        for node = 1:2
            if node == 1
                plot(wtChSet4Plot{lvl}(:,1),imag(wtChSet4Plot{lvl}(:,node)));
            else
                plot(wtChSet4Plot{lvl}(:,1),wtChSet4Plot{lvl}(:,node));
            end    
        end
    end
 end


