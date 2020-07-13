pkg"activate ."
pkg"instantiate"

using Distributions
using Plots
using MCMCChains
using StatsPlots
using Statistics
using StatsBase
##### Easy
p_grid = [x for x in range(0, length=Integer(1e4), stop=1)]
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(9, p), 6) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
N = 10000
samples = sample(p_grid,Weights(posterior), N)

#always plot!!!
plot(p_grid, posterior, seriestype = :scatter)

## 3E1
@show mapreduce(p -> p <= 0.2 ? 1 : 0, +, samples) / N
## 3E2
@show mapreduce(p -> p >= 0.8 ? 1 : 0, +, samples) / N
## 3E3
@show mapreduce(p -> (p >= 0.2 && p <= 0.8) ? 1 : 0, +, samples) / N
## 3E4
@show quantile(samples, [0.20])
## 3E5
@show quantile(samples, [0.80])
## 3E6
chn = MCMCChains.Chains(reshape(samples, N, 1, 1), ["toss"]);
MCMCChains.hpd(chn, alpha=0.33)
## 3E7
quantile(samples, [0.165, 0.835])

##### Medium

##3M1
p_grid = [x for x in range(0, length=Integer(10000000), stop=1)]
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(15, p), 8) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
#always plot!!!
plot(p_grid, posterior, seriestype = :scatter)

##3M2

N = 1000000
samples = sample(p_grid, Weights(posterior), N)

chn = MCMCChains.Chains(reshape(samples, N, 1, 1), ["toss"]);

MCMCChains.show(chn)


@show MCMCChains.hpd(chn, alpha=.1)

##3M4
p_grid = [x for x in range(0, length=Integer(1e4), stop=1)]
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(15, p), 8) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
plot(p_grid, posterior, seriestype = :scatter)

p_grid = [x for x in range(0, length=Integer(1e4), stop=1)]
likelihood = [pdf(Binomial(9, p), 6) for p in samples]
posterior = likelihood .* samples
posterior = posterior / sum(posterior)
plot(samples, posterior, seriestype = :scatter)

## 3M5
p_grid = [x for x in range(0, length=Integer(1e4), stop=1)]

prior_sq = map(p -> p<0.5 ? 0 : 0.2, p_grid)

likelihood = [pdf(Binomial(9, p), 6) for p in p_grid]
posterior = likelihood .* prior_sq
posterior = posterior / sum(posterior)
plot(p_grid, posterior)

## 3M6 (actually really hard)

# Tosses don't really mater but the likelihood(more importantly) and the priors
#the only way to modify the posterior is by modifing the prior!!
##### Hard

## 3h1

# defining the vectors

fam_1 = [1,0,0,0,1,1,0,1,0,1,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,0,
0,0,0,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0,
1,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1,0,1,1,1,1,1,0,0,1,0,1,1,0,
1,0,1,1,1,0,1,1,1,1]

fam_2 = [0,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,1,1,0,
1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,0,1,1,
0,0,0,1,1,1,0,0,0,0]

all_children = vcat(fam_1, fam_2)


# Estimating the posterior for the probability of being a boy
p_grid = [x for x in range(0, length=Integer(1e4), stop=1)]
prior = ones(length(p_grid))
likelihood =  [pdf(Binomial(200, p), sum(all_children)) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)
plot(p_grid, posterior, seriestype = :scatter)
#argument (p) that maximizes the posterior
@show p_grid[argmax(posterior)]

## 3H2
N = Integer(1e4)
samples = sample(p_grid, Weights(posterior), N)
chn = MCMCChains.Chains(reshape(samples, N, 1, 1), ["P"])
plot(chn)

# Highest posterior density intervals
#50
@show hpd(chn, alpha=0.5)
@show hpd(chn, alpha=0.11)
@show hpd(chn, alpha=0.03)

## 3H3
randombirths = rand(Binomial(200), 10000)
histogram(randombirths)
density(randombirths)

randombirths = rand(Binomial(200), 10000)

## 3H4
sum(fam_1)
randombirths = rand(Binomial(100), 10000)
histogram(randombirths)
density(randombirths)

## 3H5
randombirths_cond = rand(Binomial(49), 10000)
histogram(randombirths)
mapreduce(p -> p[1] == 1 ? p[2] : 0, +, zip(fam_1, fam_2))

# clearly there is a join probability distribution between first borns and second borns
# they are not independent!!!
