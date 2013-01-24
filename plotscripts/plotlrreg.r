eta = 0.05
lambdas = seq(0, 0.005, by=0.0005)
rmse = read.table("lrreg/rmse"); 
l2 = read.table("lrreg/l2");

plot(lambdas, l2[,1], type='l', main="Lambda vs ||w||_2, step = 0.05", xlab = "lambda", ylab="||w||_2");
plot(lambdas, rmse[,1], type='l', main = "Lambda vs Testing RMSE, step = 0.05", xlab="lambda", ylab="RMSE");
