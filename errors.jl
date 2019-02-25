using JLD
using MAT
using Dates
using Random
using Plots; pyplot()

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
triplets = Embeddings.label(data, probability_success=μ_ijk)

dimensions = 1
params = Dict{Symbol,Real}()
params[:σ] = 1/sqrt(2)
te = Embeddings.STE(triplets, dimensions, params)

@time violations = Embeddings.compute(te; max_iter=50)

Y = Embeddings.scale(data, te)
plot(data)
plot!(Y)
