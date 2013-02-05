setwd("../experiments");
l1 = read.table("lr/loss_0.001_0");
l2 = read.table("lr/loss_0.01_0");
l3 = read.table("lr/loss_0.05_0");


end = nrow(l1);
xs = (1:end) * 100;
plot(xs, l1[1: end,],  col="blue", type='l', ylim=c(0.03, 0.06), xlab="Iter", ylab="Average Loss", main="Logistic Regression Learning Curve")
lines(xs, l2[1: end, ], col='red')
lines(xs, l3[1: end, ], col='green')

text=c("step=0.001", "step=0.01", "step=0.05");
color=c("blue", "red", "green");
legend("topright", text, col=color,
       lty=1)
