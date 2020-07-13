### Intro

## code 3.1
Ppositivevampire = 0.95

Ppositivemortal = 0.01

Pvampire = 0.001

Pmortal = (1-Pvampire)

Pvampirepositive = (Ppositivevampire*Pvampire)/(Ppositivevampire*Pvampire + Ppositivemortal*Pmortal)


### Sampling from a grid-approximate posterior
## code 3.2
### doctor watson
pkg"activate ."
pkg"instantiate"


using Distributions
using Plots
using MCMCChains
using StatsPlots
using DataFrames

p_grid = range(0, step=0.001, stop=1)
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(9, p), 6) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)


## code 3.4 and 3.5


# TODO understand weights better
samples = sample(p_grid, Weights(posterior), 10000)

#density
plot(samples)
density(samples)
##code 3.4 -3.5
# In StatisticalRethinkingJulia samples are stored in an MCMCChains.Chains object.


#great example!
# TODO program HPDI with array
chn = MCMCChains.Chains(reshape(samples, N, 1, 1), ["P_of water"])

plot(chn)

## code 3.6
@show sum(map(p -> p[1]<0.5 ? p[2] : 0, zip(p_grid, posterior)))

## code 3.7
mapreduce(p -> p < 0.5 ? 1 : 0, +, samples) / N

## code 3.8
mapreduce(p -> (p > 0.5 && p < 0.75) ? 1 : 0, +, samples) / N

## code 3.9

@show quantile(samples, 0.8)

## code 3.10

@show quantile(samples, [0.1, 0.9])


## code 3.11

p_grid = range(0, step=0.001, stop=1)
prior = ones(length(p_grid))
likelihood = [pdf(Binomial(3, p), 3) for p in p_grid]
posterior = likelihood .* prior
posterior = posterior / sum(posterior)

samples = sample(p_grid, Weights(posterior), 10000);

plot(p_grid, posterior, seriestype = :scatter)

## code  3.12

chn = MCMCChains.Chains(reshape(samples, N, 1, 1), ["P_of water"]);

### here are the quantiles displayed but are only displayed for certain values
MCMCChains.show(chn)
## code  3.13

MCMCChains.hpd(chn, alpha=0.5) |> display

## code  3.14

p_grid[argmax(posterior)]

## code  3.15

maximum(samples)
## code  3.16

@show mean(samples)

@show median(samples)

## code  3.17
sum([x[2]*abs(0.5 - x[1]) for x in zip(p_grid,posterior)])

## snippet 3.18
mmp = map(p ->p[2]*abs(0.5 - p[1]),zip(p_grid,posterior))

## snippet 3.19 have to eliminate zero
p_grid[argmin(mmp)]

## snippet 3.20

Bin = Binomial(2, 0.7)

pdf(Bin, 0:2)

## snippet 3.21
pdf(Bin)

## snippet 3.22

rand(Bin, 10)

## snippet 3.23

dummy_w = rand(Binomial(2, 0.7), Integer(1e4))
zeross = count(i->(i==0), dummy_w)/1e4
oness = count(i->(i==1), dummy_w)/1e4
twoss = count(i->(i==2), dummy_w)/1e4
df = DataFrame(ONE = oness, TWO = twoss, ZERO = zeross)

## snippet 2.24

dummy_w = rand(Binomial(9, 0.7), Integer(1e4))
histogram(dummy_w, title="dummy water count")

## snippet 2.25

w = rand(Binomial(9, 0.6), Integer(1e4))

## snippet 2.26

# TODO large numbers
w = [ Binomial(9, p) for p in samples]
# Gelman what does he say
mix2 = MixtureModel(Binomial.(9, samples))
histogram(w, title="dummy water count from samples")
