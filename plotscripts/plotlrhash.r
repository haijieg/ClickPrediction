l1 = read.table("lrhashing/loss_97_");
l2 = read.table("lrhashing/loss_12289_");
l3 = read.table("lrhashing/loss_1572869_");

end = nrow(l1);
xs = (1:end) * 100;
plot(xs, l1[1: end,],  col="blue", type='l', ylim=c(0.03, 0.06), xlab="Iter", ylab="Average Loss", main="Logistic Regression Learning Curve")
lines(xs, l2[1: end, ], col='red')
lines(xs, l3[1: end, ], col='green')

text=c("step=0.001", "step=0.01", "step=0.1");
color=c("blue", "red", "green");
legend("topright", c("step=0.001", "step=0.01", "step=0.1"), col=c("black", "red", "green", "blue"),
       lty=1)
