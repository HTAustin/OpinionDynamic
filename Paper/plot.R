binary1245_X <- c(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
binary1245_Y <- c(81.38, 91.76, 92.71, 93.14, 93.34, 93.36, 93.26, 94.09, 92.81, 94.09, 94.05)

binary15_X <- c(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.0, 6.5, 7, 7.5, 8)
binary15_Y <- c(81.41, 93.83, 94.84, 95.65, 95.56, 95.61, 95.84, 96.20, 96.25, 96.38, 96.56, 96.47, 96.75, 96.65, 96.70, 96.16, 96.79)

multi_X <- c(0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)
multi_Y <- c(45.35, 56.42, 64.94, 65.00, 62.76, 66.53, 66.31, 65.83, 64.43, 65.68, 66.61)

maxY <- max(max(binary1245_Y), max(binary15_Y), max(multi_Y))
minY <- min(min(binary1245_Y), min(binary15_Y), min(multi_Y))

maxX <- max(max(binary1245_X), max(binary15_X), max(multi_X))
minX <- min(min(binary1245_X), min(binary15_X), min(multi_X))

print(maxX)
print(minX)

png("figure.png")
plot(binary1245_X, binary1245_Y, "o", pch = 16, col="blue", lwd = 2, axes = FALSE, xlim = c(0, 8.5), ylim = c(40, 100), ann = FALSE)
axis(side = 1, at = seq(0, maxX, 1))
axis(side = 2, at = seq(40, 100, 5))
lines(binary15_X, binary15_Y, "o", col="red", lwd = 2, pch = 16)
lines(multi_X, multi_Y, "o", col="green", lwd = 2, pch = 16)
box()

title(xlab = "Training data size(%)")
title(ylab = "Accuracy(%)")

title(main = "Accuracy - Training data size")

legend("bottomright", lty = 1, bty = "n", lwd = 2, legend = c("Binary 1,2,4,5", "Binary 1,5", "Multi-class"), col = c("blue", "red", "green"))

dev.off()