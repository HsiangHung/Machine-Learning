function count = accuracy(pred, y)

m = size(y,1)

count=0;
for set = 1 : m

  if (pred(set)==y(set))
     count=count+1;
  end
  
end

count=count/m
%yy =[pred,y];
   
end