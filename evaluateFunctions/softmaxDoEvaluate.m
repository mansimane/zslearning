function [ results ] = softmaxDoEvaluate( images, categories, categoryNames, theta, trainParams, doPrint, zeroCategories,outputPath )

W = stack2param(theta, trainParams.decodeInfo);
numCategories = length(categoryNames);
numImages = size(images, 2);

pred = exp(W{1}*images); % k by n matrix with all calcs needed
pred = bsxfun(@rdivide,pred,sum(pred));
[~, guessedCategories] = max(pred);

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
    guessesForCategory = guessedCategories(categories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.confusion = confusion;
results.accuracy = sum(truePos) / numImages;
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t)));

figure,
imagesc(confusion);
xticklabels(categoryNames);
yticklabels(categoryNames);
colorbar
title('Confustion Matrix post Mapping training');
file_name = [outputPath '/softmaxDoEval_conf.jpg'];
Image = getframe(gcf);
imwrite(Image.cdata, file_name);


if nargin > 6
    numUnseen = sum(arrayfun(@(x) nnz(categories == x), zeroCategories));
    results.unseenAccuracy = sum(truePos(zeroCategories)) / numUnseen;
    results.seenAccuracy = (sum(truePos) - sum(truePos(zeroCategories))) / (length(categories) - numUnseen);
end

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    if nargin > 6
        disp(['Seen accuracy: ' num2str(results.seenAccuracy)]);
        disp(['Unseen accuracy: ' num2str(results.unseenAccuracy)]);
    end
    displayConfusionMatrix(confusion, categoryNames);
end

end

