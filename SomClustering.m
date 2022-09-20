
clc
clear 
close all

gene = importdata('genes.csv');
gfeatures = gene.data;
gname = gene.textdata(2:end,1);

[m,n]= size(gfeatures);


for i=1:m
    for j=1:n
        if (isnan(gfeatures(i,j)))
            gfeatures(i,j)=0;
        end
    end
end


me = zeros(n,1);
for i=1:n
    me(i,1)= mean(gfeatures(:,i));
end
met = sum(me)/n;
for i=1:m
    for j=1:n
        if ( gfeatures(i,j)==0)
            
            gfeatures(i,j)=met;
        end
    end
end



gf = zeros(m,n);
for i=1:m
     for j=1:n
         if (gfeatures(i,j)<=-0.1)
           gf(i,j)=-1;
         end
         if (gfeatures(i,j)>=0.1)
           gf(i,j)=1;
         end
         if(gfeatures(i,j)>-0.1 && gfeatures(i,j)<0.1)
           gf(i,j)=0;
         end
     end

end


net = newsom(gfeatures',[1 4]);
net = train(net,gfeatures');

distances = dist(gfeatures, net.IW{1}');
[d,center] = min(distances,[],2);


Som =  center;

DBI = evalclusters(gfeatures,Som,'DaviesBouldin');
Sil = evalclusters(gfeatures,Som,'silhouette');
disp('DBI without discretize in SOM is:')
disp(DBI.CriterionValues);
disp('silhouette  without discretize in SOM is:');
disp(Sil.CriterionValues);

DBI2 = evalclusters(gf,Som,'DaviesBouldin');
Sil2 = evalclusters(gf,Som,'silhouette');
disp('DBI with discretize in SOM is:')
disp(DBI2.CriterionValues);
disp('silhouette  with discretize in SOM is:');
disp(Sil2.CriterionValues);
cnt=net.iw(1,:);

c1=0;
c2=0;
c3=0;
c4=0;

[m1,n1] = size(Som);
for i=1:m1
    if(Som(i)==1)
        c1=c1+1;
    end
    if(Som(i)==2)
        c2=c2+1;
    end
    if(Som(i)==3)
        c3=c3+1;
    end
    if(Som(i)==4)
        c4=c4+1;
    end
    
           
end






% nminfo = MI(cnt{1,1}(4,:),gf(3,:));
% disp(nminfo);

% xlswrite('gn.csv', gname) ;
% % cnt{1,1}(1,:) means center cluster1
% 
% xlswrite('somcenters.csv', cnt{1,1}) ;
% xlswrite('somlabels.csv', center) ;
% 
% 
% xlswrite('somdata.csv', gf);

