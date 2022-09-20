function [ names ] = ExtracteNameOfgene( geneName,label,numOfCluster)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


  for ii=1:numOfCluster
    names{ii}=cell(sum(label==ii),1);
    names{ii}=jeneName(label==ii,:);
  end
  
end

