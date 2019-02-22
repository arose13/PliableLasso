# Setup
# ~ wget http://statweb.stanford.edu/~tibs/pliable/plasso_0.1.tar.gz
# Now in R run...
# install.packages('plasso_0.1.tar.gz', repos=NULL, type='source')
# Created by: anthony
# Created on: 2/19/19
library(plasso)

# Setup
n = 200
p = 10
k = 5

x = matrix(rnorm(n*p), n, p)
z = matrix(rnorm(n*k), n, k)

y = 4*x[, 1] + 5*x[, 1] * z[, 3] + 3*rnorm(n)

# Train Model
start_time = Sys.time()
fit = plasso(x, z, y)
end_time = Sys.time()

print('=== Runtime ===')
print(end_time - start_time)

plot(fit)

print('= Model =')
print(fit)