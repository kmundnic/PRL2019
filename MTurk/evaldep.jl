include("../TripletEmbeddings.jl/src/Embeddings.jl")
include("MTurk.jl")

using MAT
using CSV
using Glob
using Plots; gr()
using Random
using Statistics
using DataFrames
using StatsPlots
using LinearAlgebra

task = "TaskA"
file = string("../data/", task, ".csv")
data = Embeddings.load_data(path=file)

path = string("/Users/karel/Documents/Research/PRL2019/green_experiment/eval_dep/", task)
if task == "TaskA"
	evaldep = CSV.read(string(path, "/eval_dep_ground_truth_1hz.csv"), allowmissing=:none)
elseif task == "TaskB"
	evaldep = CSV.read(string(path, "/eval_dep_ground_truth_10hz.csv"), allowmissing=:none)
	evaldep = evaldep[1:10:end,:]
end

# Scale and compute MSE
plot(data)
@df evaldep plot!(:Time_sec, :Data)

if task == "TaskA"
	evaldep[:Data], mse = Embeddings.scale(data[1:end-3], evaldep[:Data], MSE=true)
elseif task == "TaskB"
	evaldep[:Data], mse = Embeddings.scale(data, evaldep[:Data], MSE=true)
end

@df evaldep plot!(:Time_sec, :Data)

# Triplet violations
if task == "TaskA"
	evaldep = DataFrame(Time_sec = 1:size(data,1), Data = [evaldep[:Data]; evaldep[:Data][end]*ones(3,)])
end
files = glob(string("data/", task, "_output_*.csv"))
@assert !isempty(files) "No files found. Check path!"

# Read all files at once and concatenate the results into a single DataFrame
job = vcat(CSV.read.(files)...)
job = job[[Symbol("Input.Reference"), Symbol("Input.A"), Symbol("Input.B"), Symbol("Answer.choice")]]

job[:correct_answers] = Array{String}(undef, size(job,1))
job[Symbol("Answer.choice")] = Array{String}(undef, size(job,1))
queries = MTurk.job_queries(job)

d_ij = norm.(data[queries[:,1]] - data[queries[:,2]])
d_ik = norm.(data[queries[:,1]] - data[queries[:,3]])

for q in 1:size(queries,1)
	if d_ij[q] < d_ik[q]
		job[:correct_answers][q] = "optionA"
	elseif d_ij[q] > d_ik[q]
		job[:correct_answers][q] = "optionB"
	elseif d_ij[q] == d_ik[q]
		job[:correct_answers][q] = "optionC"
	end
end

d_ij = norm.(evaldep[:Data][queries[:,1]] - evaldep[:Data][queries[:,2]])
d_ik = norm.(evaldep[:Data][queries[:,1]] - evaldep[:Data][queries[:,3]])

for q in 1:size(queries,1)
	if d_ij[q] < d_ik[q]
		job[Symbol("Answer.choice")][q] = "optionA"
	elseif d_ij[q] > d_ik[q]
		job[Symbol("Answer.choice")][q] = "optionB"
	elseif d_ij[q] == d_ik[q]
		job[Symbol("Answer.choice")][q] = "optionC"
	end
end

violations = sum(job[Symbol("Answer.choice")] .!= job[:correct_answers])/size(job,1)
printstyled("violations = $(violations)"; color=:light_cyan)