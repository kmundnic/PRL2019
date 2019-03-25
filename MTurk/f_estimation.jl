using JLD
using MAT
using Dates
using Random
using Plots; gr()
using DataFrames
using Statistics
using LinearAlgebra

include("../TripletEmbeddings.jl/src/Embeddings.jl")
include("MTurk.jl")

Random.seed!(4)

function logistic(x; σ=20)
	return 1/(1 + exp(-σ*x))
end

function logistic_success_probabilities(data::Array{Float64}; σ=20)
	n = size(data, 1)
	D = Embeddings.distances(data, n)
	probabilities = zeros(Float64, n, n, n)

	for k = 1:n, j = 1:n, i = 1:n
		probabilities[i,j,k] = logistic(abs(D[i,j] - D[i,k]); σ=σ)
	end
	return probabilities
end

data = Embeddings.load_data(path="../data/TaskA.csv")

σ = 10
μ_ijk = logistic_success_probabilities(data; σ=σ)
job = MTurk.label_with_answers(data; probability_success=μ_ijk)
train, test = MTurk.split(job, 0.1)

deletecols!(test, Symbol("Answer.choice"))
test[Symbol("Answer.choice")] = Array{String}(undef,size(test,1))

train[:violations] = zeros(Bool, size(train, 1))
train[:violations] = train[:correct_answers] .!= train[Symbol("Answer.choice")]

dimensions = 1
params = Dict{Symbol,Real}()
params[:σ] = 1/sqrt(2)
te = Embeddings.STE(convert(Matrix{Int64}, train[:,[:i,:j,:k]]), dimensions, params)

train_violations = Embeddings.compute(te; max_iter=50)

d, μ = MTurk.success_function(job, data)
plot(d, μ, label="Empirical (triplet) errors")
plot!(d, logistic.(d), label="Model errors")

D = Embeddings.distances(te)

for t in 1:size(test,1)
	i = test[Symbol("Input.Reference")][t]
	j = test[Symbol("Input.A")][t]
	k = test[Symbol("Input.B")][t]

	if D[i,j] < D[i,k]
		test[Symbol("Answer.choice")][t] = "optionA"
	else
		test[Symbol("Answer.choice")][t] = "optionB"
	end
end

test_violations = sum(test[:correct_answers] .!= test[Symbol("Answer.choice")])/size(test,1)
@show test_violations

bins = 100
d_train, μ_train = MTurk.success_function(train, data; number_of_bins=bins)
d_test_data, μ_test_data = MTurk.success_function(test, data; number_of_bins=bins)
d_test_embedding, μ_test_embedding = MTurk.success_function(test, data; number_of_bins=bins)

plot(d_train, logistic.(d_train; σ=σ), label="True f")
plot!(d_train, μ_train, label="Estimated f from train triplets and data")
plot!(d_test_data, μ_test_data, label="Estimated f from test triplets and data")

# d_train, μ_train = MTurk.success_function(train, dropdims(Embeddings.X(te), dims=2); number_of_bins=bins)
# d_test_data, μ_test_data = MTurk.success_function(test, dropdims(Embeddings.X(te), dims=2); number_of_bins=bins)
# d_test_embedding, μ_test_embedding = MTurk.success_function(test, dropdims(Embeddings.X(te), dims=2); number_of_bins=bins)

# plot(d_train, logistic.(d_train; σ=σ), label="True f")
# plot!(d_train, μ_train, label="Estimated f from train triplets and embedding")
# plot!(d_test_data, μ_test_data, label="Estimated f from test triplets and embedding")
