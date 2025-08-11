def dotproduct (x,y):
    dotproductresult = 0
    for idx, i in enumerate(x):
        holder = int(x[idx])*int(y[idx])
        dotproductresult += holder
    print(dotproductresult) 

dotproduct(
    x=[1,0,1,5],
    y=[1,0,1,1]
)