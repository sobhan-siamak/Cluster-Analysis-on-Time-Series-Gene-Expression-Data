function DiscretData=CreateDiscretData(inputData)
    DiscretData=zeros(size(inputData));
   
    index=(inputData>0.9);
    DiscretData(index)=1;
    clear row col;

    index=(inputData<-0.9);
    DiscretData(index)=-1;
    
end