using DrWatson
@quickactivate "Statistical_rthk_Julia"

# %%
Pkg.add(["PrettyTables"])
using DataFrames
using CSV
using Distributions
using Turing
using StatsPlots

include(srcdir("quap.jl"))
include(srcdir("tools.jl"))

# %% 4.26
d = DataFrame(CSV.File(datadir("exp_raw/Howell1.csv")))
d2 = d[d.age .>= 18, :]

# %% 4.37
scatter(d2.weight, d2.height)

# %% 4.38
N = 100
a = rand(Normal(178.0, 20.0), N)
b = rand(Normal(0.0, 10.0), N)

# %% 4.39
plot(xlims = extrema(d2.weight), ylims = (-100, 400), xlable = "weight", ylabel = "height")
hline!([0.0, 272.0])
x̄ = mean(d2.weight)
x = range(minimum(d2.weight), maximum(d2.weight), length = 100)     # either
x = range(extrema(d2.weight)..., length = 100)                      # or
for (a, b) in zip(a, b)
    plot!(x, a .+ b .* (x .- x̄), color = "black", alpha = 0.2)
end
plot!(legend = false)

# %% 4.40
b = rand(LogNormal(0.0, 1.0), 10_000)
density(b, xlims = (0, 5))

# %% 4.41
N = 100
a = rand(Normal(178.0, 20.0), N)
b = rand(LogNormal(0.0, 1.0), N)

plot(xlims = extrema(d2.weight), ylims = (-100, 400), xlable = "weight", ylabel = "height")
hline!([0.0, 272.0])
x̄ = mean(d2.weight)
x = range(minimum(d2.weight), maximum(d2.weight), length = 100)
foreach(zip(a, b)) do (a, b)
    plot!(x, a .+ b .* (x .- x̄), color = "black", alpha = 0.2)
end
plot!(legend = false)

# %% 4.42
d = DataFrame(CSV.File(datadir("exp_raw/Howell1.csv")))
d2 = d[d.age .>= 18, :]
x̄ = mean(d2.weight)

@model function height(weights, heights)
    a ~ Normal(178, 20)
    b ~ LogNormal(0, 1)
    σ ~ Uniform(0, 50)
    μ = a .+ b .* (weights .- x̄)

    # This can't do the predictive posterior:
    # heights .~ Normal.(μ, σ)

    # This is pretty slow but works:
    # for i ∈ eachindex(heights)
    #     heights[i] ~ Normal(μ[i], σ)
    # end

    # This seems to work, fast:
    heights ~ MvNormal(μ, σ)
end

m4_3 = height(d2.weight, d2.height)
q4_3 = quap(m4_3, NelderMead())

# %% 4.43
@model function height_log(weights, heights)
    a ~ Normal(178, 20)
    log_b ~ Normal(0, 1)
    σ ~ Uniform(0, 50)
    μ = a .+ exp(log_b) .* (weights .- mean(weights))
    heights .~ Normal.(μ, σ)
end

m4_3b = height(d2.weight, d2.height)
q4_3b = quap(m4_3b, NelderMead())

# %% 4.44, 4.45
# precis(m4_3)

round.(q4_3.vcov, digits = 3)

# %% 4.46
scatter(d2.weight, d2.height)
post = DataFrame(rand(q4_3.distr, 10_000)', q4_3.params)
a_map = mean(post.a)
b_map = mean(post.b)
plot!(x, a_map .+ b_map .* (x .- x̄))

# %% 4.47
post = DataFrame(rand(q4_3.distr, 10_000)', q4_3.params)
post[1:5, :]

# %% 4.48, 4.49
N = 10  # rerun with 50, 150, 352
dN = d2[1:N, :]
mN = quap(height(dN.weight, dN.height))

post = rand(mN.distr, 20)
post = DataFrame(post', mN.params)

scatter(dN.weight, dN.height)
for p in eachrow(post)
    plot!(x, p.a .+ p.b .* (x .- mean(dN.weight)), color = "black", alpha = 0.3)
end
plot!(legend = false, xlabel = "weight", ylabel = "height")

# %% 4.50 - 4.52
post = rand(q4_3.distr, 1_000)
postdf = DataFrame(post', q4_3.params)
mu_at_50 = postdf.a + postdf.b * (50 - x̄)

density(mu_at_50)

quantile(mu_at_50, (0.1, 0.9))

# %% 4.53 - 4.55
weight_seq = 25:70

# It's a little unfortunate that you have to write out the formula you have already put
# into the model. I don't have a better way at the moment though.
mu = postdf.a' .+ postdf.b' .* (weight_seq .- x̄)

scatter(weight_seq, mu[:, 1:100], legend = false, c = 1, alpha = 0.1)

# %% 4.56, 4.57
mu_m = mean.(eachrow(mu))
mu_lower = quantile.(eachrow(mu), 0.055)
mu_upper = quantile.(eachrow(mu), 0.945)

scatter(d2.weight, d2.height, ms = 3)
plot!(weight_seq, mu_m, ribbon = (mu_m .- mu_lower, mu_upper .- mu_m))

# or

mu = meanlowerupper(mu)

scatter(d2.weight, d2.height, ms = 3)
plot!(weight_seq, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))

# %% 4.59 - 4.61
# This isn't really pretty either. I'm not sure how to put this into a function since
# there isn't a way to know how to create `predict_model` from `height`.
chn = Chains(post', ["a", "b", "σ"])
predict_model = height(weight_seq, missing)
sim = predict(predict_model, chn) |> Array
sim = meanlowerupper(sim')

scatter(d2.weight, d2.height, ms = 3, legend = false)
plot!(weight_seq, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
plot!(weight_seq, sim.lower, fillrange = sim.upper, alpha = 0.3, linealpha = 0.0, c = 2)

# %% 4.62
sim = predict(predict_model, Chains(rand(q4_3.distr, 10_000)', ["a", "b", "σ"])) |> Array
sim = meanlowerupper(sim')

scatter(d2.weight, d2.height, ms = 3, legend = false)
plot!(weight_seq, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
plot!(weight_seq, sim.lower, fillrange = sim.upper, alpha = 0.3, linealpha = 0.0, c = 2)

# %% 4.63
post = rand(q4_3.distr, 1_000)
postdf = DataFrame(post', q4_3.params)
weight_seq = 25:70
normals = Normal.(postdf.a' .+ postdf.b' .* (weight_seq .- x̄), postdf.σ')
sim = rand.(normals)
sim = meanlowerupper(sim)

scatter(d2.weight, d2.height, ms = 3, legend = false)
plot!(weight_seq, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
plot!(weight_seq, sim.lower, fillrange = sim.upper, alpha = 0.3, linealpha = 0.0, c = 2)



# %% 4.64
d = DataFrame(CSV.File(datadir("exp_raw/Howell1.csv")))

# %% 4.65, 6.66
d.weight_s = (d.weight .- mean(d.weight)) / std(d.weight)

f_parabola(weight_s, a, b1, b2) = a + b1 * weight_s + b2 * weight_s^2

@model function parabola(weight_s, heights)
    a ~ Normal(178, 20)
    b1 ~ LogNormal(0, 1)
    b2 ~ Normal(0, 1)
    σ ~ Uniform(0, 50)
    μ = f_parabola.(weight_s, a, b1, b2)
    heights ~ MvNormal(μ, σ)
end

q4_5 = quap(parabola(d.weight_s, d.height), NelderMead())

# precis(m4_5)

# %% 4.67
weight_seq = range(-2.2, 2, length = 30)
post = DataFrame(rand(q4_5.distr, 1_000)', q4_5.params)
mu = f_parabola.(weight_seq, post.a', post.b1', post.b2')
sim = rand.(Normal.(mu, post.σ'))

mu = meanlowerupper(mu)
sim = meanlowerupper(sim)

# %% 4.68
scatter(d.weight_s, d.height, ms = 3, alpha = 0.7, legend = false)
plot!(weight_seq, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
plot!(weight_seq, sim.lower, fillrange = sim.upper, alpha = 0.3, linealpha = 0.0, c = 2)

# %% 4.69
f_cube(weight_s, a, b1, b2, b3) = a + b1 * weight_s + b2 * weight_s^2 + b3 * weight_s^3

@model function cube(weight_s, heights)
    a ~ Normal(178, 20)
    b1 ~ LogNormal(0, 1)
    b2 ~ Normal(0, 10)
    b3 ~ Normal(0, 10)
    σ ~ Uniform(0, 50)
    μ = f_cube.(weight_s, a, b1, b2, b3)
    heights ~ MvNormal(μ, σ)
end

q4_6 = quap(cube(d.weight_s, d.height), NelderMead())

# %% 4.70, 4.71
weight_seq = range(-2.2, 2, length = 30)
post = DataFrame(rand(q4_6.distr, 1_000)', q4_6.params)
mu = f_cube.(weight_seq, post.a', post.b1', post.b2', post.b3') |> meanlowerupper
sim = rand.(Normal.(mu.raw, post.σ')) |> meanlowerupper

weight_seq_rescaled = weight_seq .* std(d.weight) .+ mean(d.weight)
scatter(d.weight, d.height, ms = 3, alpha = 0.7, legend = false)
plot!(weight_seq_rescaled, mu.mean, ribbon = (mu.mean .- mu.lower, mu.upper .- mu.mean))
plot!(weight_seq_rescaled, sim.lower, fillrange = sim.upper, alpha = 0.3, la = 0.0, c = 2)

# %% 4.72
d = DataFrame!(CSV.File(datadir("exp_raw/cherry_blossoms.csv"), missingstring = "NA"))

# precis(d)

scatter(d.year, d.doy)

# %% 4.73
d2 = dropmissing(d, :doy)

num_knots = 15
knot_list = quantile(d2.year, range(0, 1, length = num_knots))

# %% 4.74, 4.75
using BSplines: BSplineBasis, basismatrix

basis = BSplineBasis(4, knot_list)
B = basismatrix(basis, d2.year)

plot(legend = false, xlabel = "year", ylabel = "basis value")
for y in eachcol(B)
    plot!(d2.year, y)
end
plot!()

# %% 4.76
@model function spline(D, B = B)
    α ~ Normal(100, 10)
    w ~ filldist(Normal(0, 10), size(B, 2))
    σ ~ Exponential(1)
    μ = α .+ B * w
    D ~ MvNormal(μ, σ)
    return μ
end

q4_7 = quap(spline(d2.doy))

# %% 4.77
w_str = ["w[$i]" for i in 1:length(basis)]
post = DataFrame(rand(q4_7.distr, 1000)', ["α"; w_str; "σ"])

w = mean.(eachcol(post[:, w_str]))              # either
w = [mean(post[:, col]) for col in w_str]       # or

plot(legend = false, xlabel = "year", ylabel = "basis * weight")
for y in eachcol(B .* w')
    plot!(d2.year, y)
end
plot!()


# %% 4.77
w_str = ["w[$i]" for i in 1:length(basis)]
#post = DataFrame(rand(res, 1000)', ["α"; w_str; "σ"])

w = mean.(eachcol(res[:, w_str]))              # either
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

# %% 4.79
@model function spline(D, B = B)
    α ~ Normal(100, 10)
    w ~ filldist(Normal(0, 10), size(B, 2))
    σ ~ Exponential(1)
    μ = [α + sum(Brow .* w) for Brow in eachrow(B)]
    D ~ MvNormal(μ, σ)
end

q4_7alt = quap(spline(d2.doy))
