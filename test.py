def rev_func(x,length): 
   print(x[length-1],end='' '') 
   rev_func(x,length-1) 

x=[11, 12, 13, 14, 15] 
rev_func(x,5) 