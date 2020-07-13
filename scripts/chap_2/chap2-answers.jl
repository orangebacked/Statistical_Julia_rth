pkg"activate ."
pkg"instantiate"

using Distributions
using Plots
using MCMCChains
using StatsPlots
using Statistics
using StatsBase

#excersices chapter2

### 2M1
#grid aproximation 1)

p_grid = range(0, step=0.001, stop=1)

prior = ones(length(p_grid))

likelihood = [pdf(Binomial(3, p), 3) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior, seriestype = :scatter)

#grid aproximation 2)
p_grid = range(0, step=0.001, stop=1)
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(4, p), 3) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior, seriestype = :scatter)

#grid aproximation 4)
p_grid = range(0, step=0.001, stop=1)
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(7, p), 5) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior, seriestype = :scatter)

### 2M2
#squewed prior
#grid aproximation 1)
p_grid = range(0, step=0.001, stop=1)
prior_sq = map(p -> p<0.5 ? 0 : 1, p_grid)
likelihood = [pdf(Binomial(3, p), 3) for p in p_grid]
posterior = likelihood .* prior_sq
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior)

#grid aproximation 2)
p_grid = range(0, step=0.001, stop=1)
prior_sq = map(p -> p<0.5 ? 0 : 1, p_grid)
likelihood = [pdf(Binomial(4, p), 3) for p in p_grid]
posterior = likelihood .* prior_sq
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior)

#grid aproximation 4)
p_grid = range(0, step=0.001, stop=1)
prior_sq = map(p -> p<0.5 ? 0 : 1, p_grid)
likelihood = [pdf(Binomial(7, p), 5) for p in p_grid]
posterior = likelihood .* prior_sq
posterior = posterior / sum(posterior)
# ploting the posterior
plot(p_grid, posterior)
