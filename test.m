function [ accu ] = test( map, testData, state )
% This function tests the accuracy of the trained neural network according
% to the testing data.

% map is the CMAC map created using the create function. It consists of the
% input vector, the look up table, the corrected weights, and the number of
% association cells linked to each input vector.
% testData is the data to be used to test the trained CMAC.

if isempty(map) || isempty(testData)
    accu = NaN; %variable used to denote the accuracy.
    return
end

% define location of input with respect to the input vectors.
input  = zeros(length(testData),2);
for i=1:length(testData)
    if testData(i,1) > map{1}(end)
        input(i,1) = length(map{1});
    elseif testData(i,1) < map{1}(1)
        input(i,1) = 1;
    else
        temp = (length(map{1})-1)*(testData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i,1) = floor(temp);
        if (ceil(temp) ~= floor(temp)) && state
            input(i,2) = ceil(temp);
        end
    end
end

% computing the accuracy of the graph.
nume = 0;
deno = 0;
for i=1:length(input)
    if input(i,2) == 0
        output = sum(map{3}(find(map{2}(input(i,1),:))));
        nume = nume + abs(testData(i,2)-output);
        deno = deno + testData(i,2) + output;
    else
        d1 = norm(map{1}(input(i,1))-testData(i,1));
        d2 = norm(map{2}(input(i,2))-testData(i,1));
        output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
               + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
        nume = nume + abs(testData(i,2)-output);
        deno = deno + testData(i,2) + output;
    end
    Y(i) = output;
end
error = abs(nume/deno);
accu = 100 - error;

[X,I] = sort(testData(:,1));
Y = Y(I);
plot(X,Y);

% toc

end