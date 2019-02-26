using JLD
using MAT
using Dates
using Random
using Plots; gr()
using DataFrames
using Statistics
using LinearAlgebra

include("TripletEmbeddings.jl/src/Embeddings.jl")
include("Mturk/MTurk.jl")

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

data = Embeddings.load_data(path="./data/TaskA.csv")

μ_ijk = logistic_success_probabilities(data)
job = MTurk.label_with_answers(data; probability_success=μ_ijk)
train, test = MTurk.split(job, 0.1)

deletecols!(test, Symbol("Answer.choice"))
test[:predicted] = Array{String}(undef,size(test,1))

train[:violations] = zeros(Bool, size(train, 1))
train[:violations] = train[:correct_answers] .!= train[Symbol("Answer.choice")]

dimensions = 1
params = Dict{Symbol,Real}()
params[:σ] = 1/sqrt(2)
te = Embeddings.STE(convert(Matrix{Int64}, train[:,[:i,:j,:k]]), dimensions, params)

violations = Embeddings.compute(te; max_iter=50)

d, μ = MTurk.success_function(job, data)
plot(d, μ, label="Empirical (triplet) errors")
plot!(d, logistic.(d), label="Model errors")

D = Embeddings.distances(te)

for t in 1:size(test,1)
	i = test[Symbol("Input.Reference")][t]
	j = test[Symbol("Input.A")][t]
	k = test[Symbol("Input.B")][t]

	if D[i,j] < D[i,k]
		test[:predicted][t] = "optionA"
	else
		test[:predicted][t] = "optionB"
	end
end