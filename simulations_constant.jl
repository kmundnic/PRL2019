using JLD
using MAT
using Dates
using Random

include("TripletEmbeddings.jl/src/Embeddings.jl")

Random.seed!(4)

# using MAT # Need to replace for CSV or JSON format
using ArgParse

function parse_commandline()
	s = ArgParseSettings()
	@add_arg_table s begin
	    "--data", "-d"
	        help = "Path to data"
	        arg_type = String
	        required = true
	end

    return parse_args(s)
end

function mock_args() # For debugging
	args = Dict{String,Any}()
	args["data"] = "./data/TaskA.csv"

	return args
end

function success_probabilities(μ::Float64, σ::Float64, n::Int64)
    # Generate success probabilities (in the paper, these are the μ values)
    probabilities = μ .+ σ .* randn(n,n,n) # Probability of success
    probabilities[probabilities .> 1] .= 1 # Check whether any values are > 1

    return probabilities
end

function tSTE(args::Dict{String,Any}, data::Array{Float64,1}, experiment::Dict{Symbol,Any})
	# tSTE parameter
	params = Dict{Symbol,Real}()
	α = [2,10]  # Degrees of freedom of the t-Student
	n = size(data,1)

	# Initialize some variables
	mse        = zeros(Float64, length(experiment[:μ]), length(α), length(experiment[:fraction]), experiment[:repetitions])
	violations = zeros(Float64, length(experiment[:μ]), length(α), length(experiment[:fraction]), experiment[:repetitions])

	for i in 1:length(experiment[:μ])
	    # μ_ijk^a in the paper are the probabilities of successfully annotating
	    # triplets (i,j,k) by annotator a.
	    μ_ijk = success_probabilities(experiment[:μ][i], experiment[:σ], n)

	    # Generate triplets
	    triplets = Embeddings.label(data, probability_success=μ_ijk)

	    for j in 1:length(α), k in 1:length(experiment[:fraction]), l in 1:experiment[:repetitions]
			println("=========================")
			println("μ = $(experiment[:μ][i])")
			println("α = $(α[j])")
			println("fraction = $(experiment[:fraction][k]*100)%")
			println("repetition = $l")
			println("=========================")

			S = Embeddings.subset(triplets, experiment[:fraction][k]) # Random subset of triplets
			params[:α] = α[j]
	        te = Embeddings.tSTE(S, experiment[:dimensions], params)
	        @time violations[i,j,k,l] = Embeddings.compute(te; max_iter=experiment[:max_iter])

	        Y, mse[i,j,k,l] = Embeddings.scale(data, te; MSE=true)
		end
	end

	save_data(args, "tSTE", experiment, mse, violations)
end

function STE(args::Dict{String,Any}, data::Array{Float64,1}, experiment::Dict{Symbol,Any})
	# STE parameter
	params = Dict{Symbol,Real}()
	params[:σ] = 1/sqrt(2)
	n = size(data, 1)

	# Initialize some variables
	mse        = zeros(Float64, length(experiment[:μ]), length(experiment[:fraction]), experiment[:repetitions])
	violations = zeros(Float64, length(experiment[:μ]), length(experiment[:fraction]), experiment[:repetitions])

	for i in 1:length(experiment[:μ])
	    # μ_ijk^a in the paper are the probabilities of successfully annotating
	    # triplets (i,j,k) by annotator a.
	    μ_ijk = success_probabilities(experiment[:μ][i], experiment[:σ], n)

	    # Generate triplets
	    triplets = Embeddings.label(data, probability_success=μ_ijk)

	    for k in 1:length(experiment[:fraction]), l in 1:experiment[:repetitions]
			println("=========================")
			println("μ = $(experiment[:μ][i])")
			println("fraction = $(experiment[:fraction][k]*100)%")
			println("repetition = $l")
			println("=========================")

			S = Embeddings.subset(triplets, experiment[:fraction][k]) # Random subset of triplets
	        te = Embeddings.STE(S, experiment[:dimensions], params)
	        @time violations[i,k,l] = Embeddings.compute(te; max_iter=experiment[:max_iter])

	        Y, mse[i,k,l] = Embeddings.scale(data, te; MSE=true)
		end
	end

	save_data(args, "STE", experiment, mse, violations)
end

function GNMDS(args::Dict{String,Any}, data::Array{Float64,1}, experiment::Dict{Symbol,Any})
	
	n = size(data,1)

	# Initialize some variables
	mse        = zeros(Float64, length(experiment[:μ]), length(experiment[:fraction]), experiment[:repetitions])
	violations = zeros(Float64, length(experiment[:μ]), length(experiment[:fraction]), experiment[:repetitions])

	for i in 1:length(experiment[:μ])
	    # μ_ijk^a in the paper are the probabilities of successfully annotating
	    # triplets (i,j,k) by annotator a.
	    μ_ijk = success_probabilities(experiment[:μ][i], experiment[:σ], n)

	    # Generate triplets
	    triplets = Embeddings.label(data, probability_success=μ_ijk)

	    for k in 1:length(experiment[:fraction]), l in 1:experiment[:repetitions]
			println("=========================")
			println("μ = $(experiment[:μ][i])")
			println("fraction = $(experiment[:fraction][k]*100)%")
			println("repetition = $l")
			println("=========================")

			S = Embeddings.subset(triplets, experiment[:fraction][k]) # Random subset of triplets
	        te = Embeddings.HingeGNMDS(S, experiment[:dimensions])
	        @time violations[i,k,l] = Embeddings.compute(te; max_iter=experiment[:max_iter])

	        Y, mse[i,k,l] = Embeddings.scale(data, te; MSE=true)
		end
	end

	save_data(args, "GNMDS", experiment, mse, violations)

end

function CKL(args::Dict{String,Any}, data::Array{Float64,1}, experiment::Dict{Symbol,Any})
	# CKL parameter
	params = Dict{Symbol,Real}()
	μ = [2,6,10]  # μ parameter in kernel
	n = size(data,1)

	# Initialize some variables
	mse        = zeros(Float64, length(experiment[:μ]), length(μ), length(experiment[:fraction]), experiment[:repetitions])
	violations = zeros(Float64, length(experiment[:μ]), length(μ), length(experiment[:fraction]), experiment[:repetitions])

	for i in 1:length(experiment[:μ])
	    # μ_ijk^a in the paper are the probabilities of successfully annotating
	    # triplets (i,j,k) by annotator a.
	    μ_ijk = success_probabilities(experiment[:μ][i], experiment[:σ], n)

	    # Generate triplets
	    triplets = Embeddings.label(data, probability_success=μ_ijk)

	    for j in 1:length(μ), k in 1:length(experiment[:fraction]), l in 1:experiment[:repetitions]
			println("=========================")
			println("μ = $(experiment[:μ][i])")
			println("μ = $(μ[j]) (param)")
			println("fraction = $(experiment[:fraction][k]*100)%")
			println("repetition = $l")
			println("=========================")

			S = Embeddings.subset(triplets, experiment[:fraction][k]) # Random subset of triplets
			params[:μ] = μ[j]
	        te = Embeddings.CKL(S, experiment[:dimensions], params)
	        @time violations[i,j,k,l] = Embeddings.compute(te; max_iter=experiment[:max_iter])

	        Y, mse[i,j,k,l] = Embeddings.scale(data, te; MSE=true)
		end
	end

	save_data(args, "CKL", experiment, mse, violations)
end

function save_data(args::Dict{String,Any}, kind::String, experiment::Dict{Symbol,Any}, mse::Array{Float64}, violations::Array{Float64})
	# For figure generation/exploration
	folder = string("results/simulations_constant", kind, "/")
	try
		mkdir(folder)
	catch SystemError # Folder already exists
		println("Folder ", folder, " already exists")
	end

	README = string("Grid search μ = $(experiment[:μ]), fraction = $(experiment[:fraction]), repetitions = $(experiment[:repetitions]) using ", kind, " on ", 
	            Dates.format(Dates.now(), "yyyy-mm-dd_HH.MM.SS"))

    filename = string(folder, "results_", split(basename(args["data"]), ".")[1], ".mat")
	println(filename)
	matopen(filename, "w") do io
	    write(io, "mse", mse)
	    write(io, "violations", violations)
	    write(io, "README", README)
	end
end

function main()
	println("Using $(Threads.nthreads()) threads")

	# args = parse_commandline()
	args = mock_args()
	
	experiment = Dict{Symbol,Any}()
	experiment[:μ] = 0.7:0.1:0.9
	experiment[:σ] = 0.01 # Noise for triplet errors
	experiment[:dimensions] = 1
	experiment[:fraction] = 5 * 10 .^range(-4, stop=-1, length=10)[1:8] # Fraction of total number of triplets to be used to calculate the embedding ∈ [0,1]
	experiment[:repetitions] = 30
	experiment[:max_iter] = 1000

	data = Embeddings.load_data(path=args["data"])

	Random.seed!(4)
	tSTE(args, data, experiment)

	Random.seed!(4)
	STE(args, data, experiment)

	Random.seed!(4)
	GNMDS(args, data, experiment)

	Random.seed!(4)
	CKL(args, data, experiment)

end

main()