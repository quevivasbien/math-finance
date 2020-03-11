lsm <- function(X, strike=0, r=0, maxdegree=NULL) {
  if (is.null(maxdegree)) {
    maxdegree <- floor(sqrt(nrow(X)))
  }
  # start by pulling everything to its max sale value
  X <- apply(strike - X, c(1,2), max, 0)
  maxval <- max(X)
  print(maxval)
  # define variable to save regressions in
  pred_rules <- c()
  for (i in rev(1:(ncol(X)-1))) {
    # compute cash flow now
    # construct vector of future cash values
    cashnext <- rep(0, nrow(X))
    for (j in 1:nrow(X)) {
      for (k in (i+1):ncol(X)) {
        if (X[j,k] > 0) {
          cashnext[j] <-X[j,k] * (1-r)^(k-i)
          # Do I need to break here??? This may be why it doesn't seem to be working right.
        }
      }
    }
    # remove all entries where cashnow is zero since we should always continue in that case
    cashnext <- cashnext[X[,i] > 0]
    cashnow <- X[,i][X[,i] > 0]
    # estimate least squares # use 2nd-degree orthogonal polynomials for now
    degree <- min(maxdegree, length(cashnow)-1)
    if (degree < 1) {  # You really should have enough data so this never happens
      continue <- (X[,i] == 0)
      pred_rules <- c(NaN, pred_rules)
    }
    else {
      df <- data.frame(y=cashnext, x=cashnow)
      reg <- lm(y ~ poly(x, degree), data=df)
      # TODO: instead of trying to return the function, return the zeros
      f <- function(x){ x - predict(reg, data.frame(x=x)) }:
      root <- 0
      try(root <- uniroot(f, c(0, maxval^2), extendInt="upX")$root, silent=TRUE)
      pred_rules <- c(root, pred_rules)
      predictions <- X[,i]
      predictions[predictions > 0] <- predict(reg)
      continue <- (predictions >= X[,i])
    }
    # set values for future periods to zero if we don't continue
    for (j in (i+1):ncol(X)) {
      X[,j] <- X[,j] * continue
    }
    X[,i] <- X[,i] * (1-continue)  # set values to zero if we continue
  }
  return(list(cashflow_matrix=X, rules=pred_rules))  # returning as is is sloppy. find some other way!
}


gen_arima_matrix <- function(size, order=c(0,1,0), n=10) {
  t(replicate(size, arima.sim(list(order=order), n=n)))
}
