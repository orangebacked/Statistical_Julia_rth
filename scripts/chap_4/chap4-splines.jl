
#### plonomials as spleens
using DrWatson
@quickactivate "Statistical_rthk_Julia"

# using Pkg
#Pkg.add("BSplines")
using Distributions
using Plots
using MCMCChains
using Turing
using StatsPlots
using CSV
using BSplines
using DataFrames
using Statistics

include(srcdir("quap.jl"))
include(srcdir("tools.jl"))



# READ dataframe
df = DataFrame!(CSV.File(datadir("exp_raw/cherry_blossoms.csv"), missingstring = "NA"))
## code 4.72

#pythonesque funtion
d2 = dropmissing(df, :doy)           # either

# true Julia Style
#@. d2 = df[!ismissing(df.doy), :]

# other style
# d2 = d[d.doy .!== missing, :]       # or



## code 4.73
# first define the knots as quantiles
num_knots = 15

knot_list = quantile(d2.year, range(0, 1, length = num_knots))

## code 4.74

basis = BSplineBasis(4, knot_list)

B = basismatrix(basis, d2.year)

## code 4.74
plot(legend = false, xlabel = "year", ylabel = "basis value")
for y in eachcol(B)
    plot!(d2.year, y)
end
plot!()


## code 4.76


# `filldist(dist, N)` creates a multivariate distribution that is composed of `N` identical and independent copies of the univariate distribution `dist` if `dist` is univariate, or it creates a matrix-variate distribution composed of `N` identical and idependent copies of the multivariate distribution `dist` if `dist` is multivariate. `filldist(dist, N, M)` can also be used to create a matrix-variate distribution from a univariate distribution `dist`.  `arraydist(dists)` is similar to `filldist` but it takes an array of distributions `dists` as input. Writing a [custom distribution](advanced) with a custom adjoint is another option to avoid loops.

# %% 4.76 # in Turing.jl

@model function cherrymodel(D, B = B)
    α ~ Normal(100, 10)
    w ~ filldist(Normal(0, 10), size(B, 2))
    σ ~ Exponential(1)
    μ = α .+ B * w
    D ~ MvNormal(μ, σ)
end

chns_2 = sample(cherrymodel(d2.doy), NUTS(), 1000)

describe(chns_2) |> display

post = DataFrame(chns_2)
# %% 4.77
w_str = "w".*"[" .*string.(1:17).*"]"#post = DataFrame(rand(q4_7.distr, 1000)', ["α"; w_str; "σ"])

w = mean.(eachcol(post[:, w_str]))              # either
#w = [mean(post[:, col]) for col in w_str]       # or

plot(legend = false, xlabel = "year", ylabel = "basis * weight")
for y in eachcol(B .* w')
    plot!(d2.year, y)
end
plot!()


# %% 4.78
mu = post.α' .+ B * Array(post[!, w_str])'
mu = meanlowerupper(mu)

scatter(d2.year, d2.doy, alpha = 0.3)
plot!(d2.year, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
