include("MTurk.jl")
using .MTurk
using TripletEmbeddings

using MAT
using CSV
using Glob
using Plots
using Printf
using Statistics

task = "TaskA"
println("Computing embedding for ", task, " from MTurk annotations")

files = glob(string("data/", task, "_output*.csv"))
@assert !isempty(files)

mturk = vcat(CSV.read.(files)...)

queries = MTurk.job_queries(mturk)

triplets = MTurk.job_triplets(mturk)

dimensions = 1
params = Dict{Symbol,Real}()
params[:α] = 30.0
params[:σ] = 1\sqrt(2)
params[:μ] = 20

# Compute the embedding with the triplets generated in MTurk
no_triplets = floor(Int64, size(triplets,1)/2)
te = TripletEmbeddings.STE(triplets[1:no_triplets,:], dimensions, params)
@time violations = TripletEmbeddings.fit!(te; max_iter=1000)

# Load the data and scale to compute the error (MSE) and correlation between signals
# We compute the correlation because it is scale-free
data = TripletEmbeddings.load_data(string("../data/", task, ".csv"))
te.X.X[:,1], mse = TripletEmbeddings.scale(data, dropdims(TripletEmbeddings.X(te), dims=2), MSE=true)
ρ = cor(dropdims(te.X.X, dims=2), data)

println("Nº triplets = $no_triplets")
println("MSE = $(mse/length(data))")
println("Pearson ρ = $ρ")
@printf("Violations = %.2f%%\n", 100*violations)

plot(data, label="Data", color=:black)
plot!(te.X.X, label="MTurk", color=:blue)

# filename = string("results/", task, "_", string(no_triplets),  ".mat")
# file = matopen(filename, "w")
# write(file, "data", data)
# write(file, "X", te.X.X)
# write(file, "mse", mse)
# write(file, "rho", ρ)
# write(file, "violations", violations)
# close(file)