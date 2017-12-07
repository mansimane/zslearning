function [ guessedCategories, results ] = evaluateGaussianBayesian(thetaSeenSoftmax, thetaUnseenSoftmax, ...
    thetaMapping, seenSmTrainParams, unseenSmTrainParams, mapTrainParams, images, ...
    categories, cutoffs, zeroCategoryTypes, nonZeroCategoryTypes, categoryNames, wordVectors,mappedTestImages, doPrint)

addpath toolbox;

numImages = size(images, 2);
numCategories = length(zeroCategoryTypes) + length(nonZeroCategoryTypes);
Ws = stack2param(thetaSeenSoftmax, seenSmTrainParams.decodeInfo);
Wu = stack2param(thetaUnseenSoftmax, unseenSmTrainParams.decodeInfo);

%mappedImages = mapDoMap(images, thetaMapping{1}, mapTrainParams);
%mappedImages = mappedTestImages;
% This is the seen label classifier
probSeen = exp(Ws{1}*images); % k by n matrix with all calcs needed
probSeen = bsxfun(@rdivide,probSeen,sum(probSeen));
probSeenFull = zeros(numCategories, numImages);
probSeenFull(nonZeroCategoryTypes, :) = probSeen;

% This is the unseen label classifier
%probUnseenFull is 10
pz1 = zeros(length(nonZeroCategoryTypes),numImages);
pz2 = zeros(length(nonZeroCategoryTypes),numImages);
probUnseenFull = zeros(numCategories, numImages);

for i=1:length(nonZeroCategoryTypes)
    probUnseen = exp(Wu{1}*mappedTestImages{i}); % 2 by n matrix with all calcs needed
    pz1(i,:) = probUnseen(1,:);
    pz2(i,:) = probUnseen(2,:);
end
pz1 = bsxfun(@rdivide,pz1,sum(pz1));
pz2 = bsxfun(@rdivide,pz2,sum(pz2));

pz1_max = max(pz1);%max across rows
pz2_max = max(pz2);
probUnseen_max = [max(pz1);max(pz2)];
probUnseenFull(zeroCategoryTypes, :) = probUnseen_max;

% Treat everything as unseen first, then filter out cases
% where things fall outside cutoff circles
probs = ones(size(categories));
for c_i = 1:length(nonZeroCategoryTypes)
    currentCategory = nonZeroCategoryTypes(c_i);
    centerVector = wordVectors(:, currentCategory);
    dists = slmetric_pw(centerVector, mappedTestImages{c_i}, 'eucdist');
    probs(dists < cutoffs(currentCategory)) = 0; % falls in circle; is not unseen    
end

%probs = 0, Seen. probs =1 unseen
finalProbs = bsxfun(@times, probSeenFull, 1 - probs') + bsxfun(@times, probUnseenFull, probs');
[~, guessedCategories ] = max(finalProbs);

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
numUnseen = sum(arrayfun(@(x) nnz(categories == x), zeroCategoryTypes));
results.unseenAccuracy = sum(truePos(zeroCategoryTypes)) / numUnseen;
results.seenAccuracy = (sum(truePos) - sum(truePos(zeroCategoryTypes))) / (numImages - numUnseen);
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));
results.confusion = confusion;

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Seen Accuracy: ' num2str(results.seenAccuracy)]);
    disp(['Unseen Accuracy: ' num2str(results.unseenAccuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, categoryNames);
end

end

