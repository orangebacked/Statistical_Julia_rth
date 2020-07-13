### snippet 4.1


### always use!!!! got procken packages once and never again sh
pkg"activate ."
pkg"instantiate"
# using Pkg
# Pkg.add("StatisticalRethinking")
Pkg.add("Plots")
Pkg.add("Turing")
Pkg.add("Distributions")
# Pkg.add("Plots")
# Pkg.add("Optim")
Pkg.add("CSV")
Pkg.add("MCMCChains")
# Pkg.add("Gadfly")
Pkg.add("StatsPlots")
using Distributions
using Plots
# using Optim
using MCMCChains
using Turing
using StatsPlots
using CSV
# using StatisticalRethinking
# No attempt has been made to condense this too fewer lines of code

noofsteps = 20;
noofwalks = 15;
pos = Array{Float64, 2}(rand(Uniform(-1, 1), noofsteps, noofwalks));
pos[1, :] = zeros(noofwalks);
csum = cumsum(pos, dims=1);

f = Plots.font("DejaVu Sans", 6)
mx = minimum(csum) * 0.9

# Plot and annotate the random walks

p1 = plot(csum, leg=false, title="Random walks ($(noofwalks))")
plot!(p1, csum[:, Int(floor(noofwalks/2))], leg=false, title="Random walks ($(noofwalks))", color=:black)
plot!(p1, [5], seriestype="vline")
annotate!(5, mx, text("step 4", f, :left))
plot!(p1, [9], seriestype="vline")
annotate!(9, mx, text("step 8", f, :left))
plot!(p1, [17], seriestype="vline")
annotate!(17, mx, text("step 16", f, :left))

# densities
p2 = Vector{Plots.Plot{Plots.GRBackend}}(undef, 3);
plt = 1
for step in [4, 8, 16]
  indx = step + 1 # We aadded the first line of zeros
  global plt
  global fitl = fit_mle(Normal, csum[indx, :])
  lx = (fitl.μ-4*fitl.σ):0.01:(fitl.μ+4*fitl.σ)
  p2[plt] = density(csum[indx, :], legend=false, title="$(step) steps")
  plot!( p2[plt], lx, pdf.(Normal( fitl.μ , fitl.σ ) , lx ), fill=(0, .5,:orange))
  plt += 1
end
p3 = plot(p2..., layout=(1, 3))
plot(p1, p3, layout=(2,1))


### snippet 4.2

prod(1 .+ rand(Uniform(0, 0.1), 12))

### snippet 4.3

growth = [prod(1 .+ rand(Uniform(0, 0.1), 10)) for i in 1:10000];

fitl = fit_mle(Normal, growth)
plot(Normal(fitl.μ , fitl.σ ), fill=(0, .5,:orange), lab="Normal distribution")
density(growth, lab="'sample' distribution")


### snippet 4.4

#big_growth

big = [prod(1 .+ rand(Uniform(0, 0.5), 12)) for i in 1:10000];
small = [prod(1 .+ rand(Uniform(0, 0.01), 12)) for i in 1:10000];
fitb = fit_mle(Normal, big)
fits = fit_mle(Normal, small)
p1 = plot(Normal(fitb.μ , fitb.σ ), lab="Big normal distribution", fill=(0, .5,:orange))
p2 = plot(Normal(fits.μ , fits.σ ), lab="Small normal distribution", fill=(0, .5,:orange))
density!(p1, big, lab="'big' distribution")
density!(p2, small, lab="'small' distribution")


### snippet 4.5

logbig = log.(big)
fitlogbig = fit_mle(Normal, logbig)
p3 = plot(Normal(fitlogbig.μ , fitlogbig.σ ), lab="Big normal distribution", fill=(0, .5,:orange))
density!(p3, logbig, lab="'log' distribution")
plot(p1, p2, p3,layout=(1, 3))

## code 4.6

using Distributions
using Plots
# using StatisticalRethinking
using Optim
using MCMCChains
using StatsPlots
using CSV

p_grid = [x for x in range(0, length=100, stop=1)]
Posterior1 = [pdf(Binomial(9, x),6) for x in p_grid]
num = Posterior1.*p_grid
den = sum()



### first golem implemented with Turing

# code 4.7
ProjDir = @__DIR__
df = CSV.read("/home/orangebacked/Documents/statistical-rethinking/StatisticalRethinking.jl/data/Howell1.csv")

# code 4.8
@show df

# code 4.9
df2 = filter(row -> row[:age] >= 18, df);

#show density (kinda normal)

fitlogbig = fit_mle(Normal, df2.height)
plot(Normal(fitlogbig.μ , fitlogbig.σ ), lab="Big normal distribution", fill=(0, .5,:orange))
density!(df2.height)


# code 4.11
p = Vector{Plots.Plot{Plots.GRBackend}}(undef, 4)

d1 = Normal(178, 20)
p[1] = plot(100:250, [pdf(d1, μ) for μ in 100:250],
	xlab="μ",
	ylab="density",
	lab="Prior on μ")

# code 4.13

d2 = Uniform(0,50)
p[2] = plot(-10:60, [pdf(d2, σ) for σ in -10:60],
	xlab="sigma",
	ylab="density",
	lab="Prior on sigma")

# density of a uniform
density(rand(Uniform(0,50), 100000))

# density of a normal
density(rand(Normal(178, 20), 100000))

#### basically all the height model 4.14 -4.30
using Turing
using StatsPlots

@model line(y) = begin
    #priors
    μ ~ Normal(178, 20)
    σ ~ Uniform(0, 50)

    #model
		N = length(y)
		for n in 1:N
    	y[n] ~ Normal.(μ, σ)
		end
end

chns = sample(line(df2.height), NUTS(), 10000)

μ_summary = chns[:μ]

plot(μ_summary, seriestype = :histogram)

σ_summary = chns[:σ]

plot(σ_summary, seriestype = :histogram)

describe(chns) |> display

μ_array = get(chns, :μ)[1].data

σ_array = get(chns, :σ)[1].data

MCMCChains.hpd(chns, alpha=0.2)

arr = vcat(transpose.(vecs)...)'
cor(arr)

########### 4.30 - 4.50
### linear regression

## code 4.42
using Turing
using StatsPlots
using Plots

height = df2.height

weight = df2.weight

@model heightmodel(y, x) = begin
    #priors
    α ~ Normal(178, 100)
	σ ~ Uniform(0, 50)
	β ~ LogNormal(0, 10)

	x_bar = mean(x)
    #model

		μ = α .+ (x.-x_bar).*β
    y ~ MvNormal(μ, σ)
end

chns = sample(heightmodel(height, weight), NUTS(), 100000)

## code 4.43
describe(chns) |> display


## HPDI
MCMCChains.hpd(chns, alpha=0.11)

# covariance and correlation

alph = get(chns, :α)[1].data

bet = get(chns, :β)[1].data

sigm = get(chns, :σ)[1].data

vecs = (alph, bet, sigm)

arr = vcat(transpose.(vecs)...)'
cov(arr)
cor(arr)

ones = mean(alph)
twos = mean(bet)

ff(x) = ones + twos.*(x-50)


## 4.46
plot(weight,height, seriestype = :scatter)
plot!(ff, 30, 60)
vecs = (alph[1:99000], bet1[1:99000], bet2[1:99000])
arr = vcat(transpose.(vecs)...)'

polinomial = [vec(alph + bet1.*(x) + bet2.*(x.^2)) for x in -2:0.01:2]

arrr = vcat(transpose.(polinomial)...)'

plot([mean(arrr[:,x]) for x in 1:401],-2:0.01:2)

plot([mean(arrr[:,x]) for x in 1:401],-2:0.01:2, ribbon = ([-1*(quantile(arrr[:,x],[0.1,0.9])[1] - mean(arrr[:,x])) for x in 1:46], [quantile(arrr[:,x],[0.1,0.9])[2] - mean(arrr[:,x]) for x in 1:46]))

### subpolots

p = plot(weight, height, seriestype = :scatter)

for n in 1:100
	println(n)
	o = alph[n]
	s = bet[n]
	fff(x) = o + s*(x-50)
	plot!(p, fff, 30, 60)
end

p



quantile(height, [0.05, 0.89])


### μ att 50


chns = sample(heightmodel(height, weight), NUTS(), 10000)

alph = get(chns, :α)[1].data

bet = get(chns, :β)[1].data

sigm = get(chns, :σ)[1].data

μ_at = alph[1:352] .+ bet[1:352].*(weight .- mean(weight))

density(μ_at)

quantile(μ_at, [0.1, 0.9])

## quantifing posterior for each x this is sooo important
vecs = (alph[1:352], bet[1:352])
arr = vcat(transpose.(vecs)...)'

ss = [vec(alph + bet.*(x)) for x in 25:1:70]

arrr = vcat(transpose.(ss)...)'

plot([mean(arrr[:,x]) for x in 1:46],25:1:70, ribbon = ([-1*(quantile(arrr[:,x],[0.1,0.9])[1] - mean(arrr[:,x])) for x in 1:46], [quantile(arrr[:,x],[0.1,0.9])[2] - mean(arrr[:,x]) for x in 1:46]))


### distribution of the height


### matrix of distributions this gets stranger and stranger
ffff = [rand(Normal(vec(alph + bet.*x)[n], sigm[n])) for n in 1:9000, x in 25:1:70]

quantile(ffff[:,1],[0.1,0.9])

plot!([mean(arrr[:,x]) for x in 1:46],25:1:70, ribbon = ([-1*(quantile(ffff[:,x],[0.1,0.9])[1] - mean(arrr[:,x])) for x in 1:46], [quantile(ffff[:,x],[0.1,0.9])[2] - mean(arrr[:,x]) for x in 1:46]))
