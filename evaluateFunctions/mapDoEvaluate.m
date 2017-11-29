function [ guessedCategoriesDebug, results ] = mapDoEvaluate( images, categories, cat_id, originalCategoryNames, testCategoryNames, testWordTable, theta, trainParams, doPrint )

numImages = size(images, 2);
numCategories = size(testWordTable, 2);

% Feedforward
mappedImages = mapDoMap(images, theta{1}, trainParams);

dist = slmetric_pw(testWordTable, mappedImages, 'eucdist');
[ ~, guessedCategories ] = min(dist);

% map categories from originalCategoryNames to testCategoryNames
mappedCategorySet = zeros(1, length(originalCategoryNames));
for i = 1:length(originalCategoryNames)
    mappedCategorySet(i) = find(strcmp(originalCategoryNames{i}, testCategoryNames));
end
mappedCategories = arrayfun(@(x) mappedCategorySet(x), categories);

guessedCategoriesDebug = [ dist; mappedCategories'; guessedCategories ];

% Calculate scores
confusion = zeros(numCategories, numCategories);
for actual = 1:numCategories
	%how many labels are predicted as 1..etc
    guessesForCategory = guessedCategories(mappedCategories == actual);
    for guessed = 1:numCategories
        confusion(actual, guessed) = sum(guessesForCategory == guessed);
    end
end

truePos = diag(confusion); % true positives, column vector
results.accuracy = sum(truePos) / numImages;
t = truePos ./ sum(confusion, 2);
results.avgPrecision = mean(t(isfinite(t), :));
t = truePos' ./ sum(confusion, 1);
results.avgRecall = mean(t(:, isfinite(t))); %Filter out NaNs if present

if doPrint == true
    disp(['Accuracy: ' num2str(results.accuracy)]);
    disp(['Averaged precision: ' num2str(results.avgPrecision)]);
    disp(['Averaged recall: ' num2str(results.avgRecall)]);
    displayConfusionMatrix(confusion, testCategoryNames);
end

end

