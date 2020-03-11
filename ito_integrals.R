n_sims <- 10000
n_divs <- 1000

integrals <- c()
delta <- 1/n_divs
ticks <- seq(0, 1, delta)
stdev <- sqrt(delta)

for (i in 1:n_sims) {
  B <- cumsum(c(0, rnorm(n_divs, sd=stdev)))
  integral <- 0
  for (j in 1:n_divs) {
    integral <- integral + B[j]*(B[j+1] - B[j])
  }
  integrals[i] <- integral
}

# (integrals + 0.5)*2 should be approximately chisq(1) distributed (see Oksendal p. 26)