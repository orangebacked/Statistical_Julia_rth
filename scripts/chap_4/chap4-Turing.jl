#### plonomials as spleens
pkg"activate ."
pkg"instantiate"


using Distributions
using Plots
# using Optim
using MCMCChains
using Turing
using StatsPlots
using CSV


ProjDir = @__DIR__
df = CSV.read("/home/orangebacked/Documents/statistical-rethinking/StatisticalRethinking.jl/data/Howell1.csv")

# code 4.8
@show df

# plotting the relationship
df2 = filter(row -> row[:age] >= 18, df);

plot([(df.height[x], df.weight[x]) for x in 1:354], seriestype = :scatter)



## code 4.65
height = df.height

weight_s = (df.weight .-mean(df.weight))./std(df.weight)

weight_s² = weight_s.^2

@model heightmodel(height, weight, weight²) = begin
    #priors
    α ~ Normal(178, 20)
	σ ~ Uniform(0, 50)
	β1 ~ LogNormal(0, 1)
	β2 ~ Normal(0, 1)
    #model
	μ = α .+ weight.*β1 + weight².*β2
    height ~ MvNormal(μ, σ)
end

chns = sample(heightmodel(height, weight_s, weight_s²), NUTS(), 100000)

describe(chns) |> display

### painting the fit

alph = get(chns, :α)[1].data

bet1 = get(chns, :β1)[1].data

bet2 = get(chns, :β2)[1].data


# plot for the data but we must look at the data that has been transformed

## code 4.67 and code 4.68
# This is the mean
plot([(weight_s[x], df.height[x]) for x in 1:354], seriestype = :scatter)

f_s(x) = mean(alph) + mean(bet1)*x + mean(bet2)*(x^2)

plot!(f_s, -2, 3)

# This is the posterior plot TODO it must be fixed

vecs = (alph[1:99000], bet1[1:99000], bet2[1:99000])
arr = vcat(transpose.(vecs)...)'

polinomial = [vec(alph + bet1.*(x) + bet2.*(x.^2)) for x in -2:0.01:2]

arrr = vcat(transpose.(polinomial)...)'

plot([mean(arrr[:,x]) for x in 1:401],-2:0.01:2)

plot([mean(arrr[:,x]) for x in 1:401],-2:0.01:2, ribbon = ([-1*(quantile(arrr[:,x],[0.1,0.9])[1] - mean(arrr[:,x])) for x in 1:46], [quantile(arrr[:,x],[0.1,0.9])[2] - mean(arrr[:,x]) for x in 1:46]))


## code 4.69


height = df.height

weight_s = (df.weight .-mean(df.weight))./std(df.weight)

weight_s² = weight_s.^2

weight_s³ = weight_s.^3

@model heightmodel(height, weight, weight², weight³) = begin
    #priors
    α ~ Normal(178, 20)
	σ ~ Uniform(0, 50)
	β1 ~ LogNormal(0, 1)
	β2 ~ Normal(0, 1)
	β3 ~ Normal(0, 1)
    #model
	μ = α .+ weight.*β1 + weight².*β2 + weight³.*β3
    height ~ MvNormal(μ, σ)
end

chns_2 = sample(heightmodel(height, weight_s, weight_s², weight_s³), NUTS(), 100000)

describe(chns_2) |> display

### painting the fit

alph = get(chns, :α)[1].data

bet1 = get(chns, :β1)[1].data

bet2 = get(chns, :β2)[1].data
