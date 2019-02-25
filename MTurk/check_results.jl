include("../TripletEmbeddings.jl/src/Embeddings.jl")
include("MTurk.jl")

using MAT
using CSV
using Glob
using Plots; pyplot()
using Random
using DataFrames
using LinearAlgebra
using DataStructures

Random.seed!(10)

task = "TaskB"
println("Checking annotations for ", task)

file = string("../data/", task, ".csv")
data = Embeddings.load_data(path=file)

files = glob(string("data/", task, "_output_*.csv"))
@assert !isempty(files) "No files found. Check path!"

# Read all files at once and concatenate the results into a single DataFrame
mturk = vcat(CSV.read.(files)...)
mturk = mturk[[:WorkerId, Symbol("Input.Reference"), Symbol("Input.A"), Symbol("Input.B"), Symbol("Answer.choice")]]

hits = MTurk.hits_per_worker(convert(Array{String,1}, mturk[:WorkerId]))

no_violations = OrderedDict{String, Int}()
optionA = OrderedDict{String, Int}()
optionB = OrderedDict{String, Int}()
optionC = OrderedDict{String, Int}()

for worker in keys(hits)
	job = mturk[mturk[:,:WorkerId] .== worker, :]

	optionA[worker] = count(x -> x .== "optionA", job[Symbol("Answer.choice")])
	optionB[worker] = count(x -> x .== "optionB", job[Symbol("Answer.choice")])

	no_violations[worker] = MTurk.job_violations(job, data)

	# @show (worker, no_violations//hits[worker])
end

results = DataFrame(workerID = collect(keys(optionA)),
		  		optionA = collect(values(optionA)),
		  		optionB = collect(values(optionB)),
		  		violations = collect(values(no_violations)),
		  		total = collect(values(hits)))

mturk[:correct_answers] = Array{String}(undef, size(mturk,1))
queries = MTurk.job_queries(mturk)

d_ij = norm.(data[queries[:,1]] - data[queries[:,2]])
d_ik = norm.(data[queries[:,1]] - data[queries[:,3]])

for q in 1:size(queries,1)
	if d_ij[q] < d_ik[q]
		mturk[:correct_answers][q] = "optionA"
	elseif d_ij[q] > d_ik[q]
		mturk[:correct_answers][q] = "optionB"
	elseif d_ij[q] == d_ik[q]
		mturk[:correct_answers][q] = "optionC"
	end
end

results[:fraction_correct] = zeros(Float64, size(results,1))
results[:fraction_correct] = 1 .- results[:violations]./results[:total]

### Analysis of errors
# This is done by
if task == "TaskA"
	annotators = ["A2QLSHXNCHBRN4", "A14I3K8UN3612X", "A1T79J0XQXDDGC"] # (first, second, third top annotators)
elseif task == "TaskB"
	annotators = ["A2MCG5W6LHSRG9", "A27PBC5O3Z5ZED", "A2ZV821LZHVOHD"] # (first, second, third top annotators)
end

bins = 10

μ = zeros(Float64, size(annotators,1), bins - 1)
d = zeros(Float64, size(annotators,1), bins - 1)

plot()
for a in eachindex(annotators)
	job = mturk[mturk[:,:WorkerId] .== annotators[a], :] # First annotator

	d[a,:], μ[a,:] = MTurk.success_function(job, data; number_of_bins=bins)

	plot!(d[a,:], μ[a,:],
		xlabel="d_ik - d_ij",
		ylabel="Probability of success", 
		show=true)
end
plot!(mean(d, dims=1)', mean(μ, dims=1)')

# filename = string(task, "_annotators", ".mat")
# file = matopen(filename, "w")
# write(file, "d", d)
# write(file, "mu", μ)
# close(file)