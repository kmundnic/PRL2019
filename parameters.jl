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

function main()
	println("Using $(Threads.nthreads()) threads")

	# args = parse_commandline()
	args = mock_args()

	data = Embeddings.load_data(path=args["data"])

	# Model parameters
	params = Dict{Symbol,Real}()
	α = [2,10]  # Degrees of freedom of the t-Student
	fraction = 5 * 10 .^range(-4, stop=-1, length=10)[1:8] # Fraction of total number of triplets to be used to calculate the embedding ∈ [0,1]
	dimensions = 1 # dimensions of the embedding

	# Probability parameters
	μ = 0.70:0.1:0.8
	σ = 0.01
	n = size(data,1)

	# Repetitions
	repetitions = 2 # To take mean values and compare

	# Initialize some variables
	mse        = zeros(Float64, length(μ), length(α), length(fraction), repetitions)
	violations = zeros(Float64, length(μ), length(α), length(fraction), repetitions)

	n = size(data,1)

	for i in 1:length(μ)
	    # μ_ijk^a in the paper are the probabilities of successfully annotating
	    # triplets (i,j,k) by annotator a.
	    μ_ijk = success_probabilities(μ[i], σ, n)

	    # Generate triplets
	    triplets = Embeddings.label(data, probability_success=μ_ijk)

	    for j in 1:length(α), k in 1:length(fraction), l in 1:repetitions
			println("=========================")
			println("μ = $(μ[i])")
			println("α = $(α[j])")
			println("fraction = $(fraction[k])")
			println("repetition = $l")
			println("=========================")

			S = Embeddings.subset(triplets, fraction[k]) # Random subset of triplets
			params[:α] = α[j]

	        te = Embeddings.tSTE(S, dimensions, params)
	        @time violations[i,j,k,l] = Embeddings.compute(te; max_iter=50)

	        Y, mse[i,j,k,l] = Embeddings.scale(data, te; MSE=true)
		end
	end

	# For figure generation/exploration
	try
		mkdir("results/")
	catch SystemError # Folder already exists
		println("Folder results/ already exists")
	end

	README = string("Grid search μ = $(μ), α = $(α), fraction = $(fraction), repetitions = $repetitions using t-STE on ", 
	            Dates.format(Dates.now(), "yyyy-mm-dd_HH.MM.SS"))

    filename = string("./results/results_", split(basename(args["data"]), ".")[1], ".mat")
	println(filename)
	matopen(filename, "w") do io
	    write(io, "mse", mse)
	    write(io, "violations", violations)
	    write(io, "README", README)
	end
end

main()