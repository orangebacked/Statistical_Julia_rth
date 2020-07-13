pkg"activate ."
pkg"instantiate"

using Distributions
using Plots
using StatisticalRethinking

# grid aproximation

function plotfort(t, n, s)

    bn(x) = pdf(Binomial(t, x),s)

    x = rand(n)

    prior = ones(n)

    likelyhoaod = bn.(x)


    post = similar(x)

    post .= prior.*likelyhoaod

    posterior = similar(x)

    posterior .= post./(sum(post))

    plot(x, posterior, seriestype = :scatter)

end

plotfort(7, 10000, 5)

t = 7
n = 1000
s = 5

bn(x) = pdf(Binomial(t, x),s)

x = [x for x in range(0, length=n, stop=1)]

# plot(x, seriestype = :scatter)

prior = ones(n)

likelyhoaod = bn.(x)

post = similar(x)

post .= prior.*likelyhoaod

posterior = similar(x)

posterior .= post./(sum(post))

plot(x, posterior, seriestype = :scatter)

using StatisticalRethinking


N = 10000
samples = sample(x, Weights(posterior), N)
fitnormal= fit_mle(Normal, samples)

sample(p) = pdf(fitnormal, p)

likelyhoasdod = sample.(x)

##### quadratic aprox


x0 = [0.5]
lower = [0.0]
upper = [1.0]

function loglik(x)
  ll = 0.0
  ll += log.(pdf.(Beta(1, 1), x[1]))
  ll += sum(log.(pdf.(Binomial(9, x[1]), repeat([6], 1))))
  -ll
end

opt = optimize(loglik, lower, upper, x0, Fminbox(GradientDescent()))
qmap = Optim.minimizer(opt)
