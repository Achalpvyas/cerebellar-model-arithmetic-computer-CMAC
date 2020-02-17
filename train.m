function [ map, ite, finalError, t ] = train( parentMap, trainData, E, state )
% The traning function is used to train the neural network in accordance with the training data.
% Which means that it provides the corrected weights to the CMAC architecture.
% The CMAC map is created making use of the create function. It includes the input vector, 
% the look up table, the initial weights, and the number of
% associated cells linked to all the input vector.
% TrainData is used to train CMAC.
% E is the acceptable error, which is in the terms of the training data.
% state is the measure which helps to states CMAC into discrete or continuous.

tic;

map = parentMap;
if isempty(map) || isempty(trainData) || isempty(E)
    return
end

% It defines the location of input with respect to the input vectors
input  = zeros(length(trainData),2);
for i=1:length(trainData)
    if trainData(i,1) > map{1}(end)
        input(i,1) = length(map{1});
    elseif trainData(i,1) < map{1}(1)
        input(i,1) = 1;
    else
        temp = (length(map{1})-1)*(trainData(i,1)-map{1}(1))/(map{1}(end)-map{1}(1)) + 1;
        input(i,1) = floor(temp);
        if (ceil(temp) ~= floor(temp)) && state
            input(i,2) = ceil(temp);
        end
    end
end

% Now compute the output of each input and respectively adjust weights until the
% specified number of iterations are obtained.
eta = 0.025; % eta is the learning rate
error = Inf;
ite = 0; %variable used for keeping track of number of itrations.
z = 0; % Variable to keep a track of count.
while (error > E)&&(2*z <= ite)
    old_err = error;
    ite = ite + 1;
    
    % compute output for each input and respectively adjust the weights.
    for i=1:length(input)
        if input(i,2) == 0
            output = sum(map{3}(find(map{2}(input(i,1),:))));
            error = eta*(trainData(i,2)-output)/map{4};
            map{3}(find(map{2}(input(i,1),:))) = map{3}(find(map{2}(input(i,1),:))) + error;
        else
            d1 = norm(map{1}(input(i,1))-trainData(i,1));
            d2 = norm(map{1}(input(i,2))-trainData(i,1));
            output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
                    + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
            error = eta*(trainData(i,2)-output)/map{4};
            map{3}(find(map{2}(input(i,1),:))) = map{3}(find(map{2}(input(i,1),:)))...
                                                    + (d2/(d1+d2))*error;
            map{3}(find(map{2}(input(i,2),:))) = map{3}(find(map{2}(input(i,2),:)))...
                                                    + (d1/(d1+d2))*error;            
        end
    end

    % Now computing the final error
    nume = 0;
    deno = 0;
    for i=1:length(input)
        if input(i,2) == 0
            output = sum(map{3}(find(map{2}(input(i,1),:))));
            nume = nume + abs(trainData(i,2)-output);
            deno = deno + trainData(i,2) + output;
        else
            d1 = norm(map{1}(input(i,1))-trainData(i,1));
            d2 = norm(map{1}(input(i,2))-trainData(i,1));
            output = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
                   + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
            nume = nume + abs(trainData(i,2)-output);
            deno = deno + trainData(i,2) + output;
        end
    end
    error = abs(nume/deno);
    if abs(old_err - error) < 0.00001
        z = z + 1;
    else
        z = 0;
    end
end
ite = ite - z;

% Now compute the final error
nume = 0;
deno = 0;
for i=1:length(input)
    if input(i,2) == 0
        Y(i) = sum(map{3}(find(map{2}(input(i,1),:))));
        nume = nume + abs(trainData(i,2)-Y(i));
        deno = deno + trainData(i,2) + Y(i);
    else
        d1 = norm(map{1}(input(i,1))-trainData(i,1));
        d2 = norm(map{1}(input(i,2))-trainData(i,1));
        Y(i) = (d2/(d1+d2))*sum(map{3}(find(map{2}(input(i,1),:))))...
               + (d1/(d1+d2))*sum(map{3}(find(map{2}(input(i,2),:))));
        nume = nume + abs(trainData(i,2)-Y(i));
        deno = deno + trainData(i,2) + Y(i);
    end
end
finalError = abs(nume/deno);
[X,I] = sort(trainData(:,1));
Y = Y(I);
%plot(X,Y);

t = toc;

end