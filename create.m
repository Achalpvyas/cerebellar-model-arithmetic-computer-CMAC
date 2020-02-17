function [ map ] = create( X, numWeights, numCell )
% This function will link all the input vector to the association cells and
% will assign all weights to one.
% Here 'X' represents the input values.
% numWeights represents the number of weights in other words the hidden 
% vectors to be used in the CMAC algorithm
% numCell represents the number of Association cells to be linked with each
% input vector.
% It's important to be known that, numCell option means that, overlap
% between the successive input vectors is equal to numCell-1.

if (numCell > numWeights) || (numCell < 1) || (isempty(X))
    map = [];
    return
end

x = linspace(min(X),max(X),numWeights-numCell+1)';

LUT = zeros(length(x),numWeights);
for i=1:length(x)
    LUT(i,i:numCell+i-1) = 1;
end

W = ones(numWeights,1);

map = cell(3,1);
map{1} = x;
map{2} = LUT;
map{3} = W;
map{4} = numCell;

end