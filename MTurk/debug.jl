include("../../Embeddings/src/utilities.jl")
include("MTurk.jl")

using MAT
using CSV
using Glob
using Plots; pyplot()
using DataFrames
using DataStructures

srand(10)

task = "TaskB"
println("Checking annotations for ", task)

file = string("../data/", task, ".csv")
data = load_data(path=file)

files = glob(string(task, "_output_*.csv"))

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

	# # If we get a correct answer as optionC, we make a choice
	# # If we don't do this, it is unfair for annotators
	# # We randomly select an option between A & B
	# for i in 1:size(job,1)
	# 	if job[:correct_answers][i] == "optionC"
	# 		job[:correct_answers][i] = rand(["optionA", "optionB"])
	# 	end
	# end

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

mturk[:correct_answers] = Array{String,1}(size(mturk,1))
queries = MTurk.job_queries(mturk)

d_ij = norm.(data[queries[:,1]] - data[queries[:,2]])
d_ik = norm.(data[queries[:,1]] - data[queries[:,3]])

for q in 1:size(queries,1)
	if d_ij[q] < d_ik[q]
		mturk[:correct_answers][q] = "optionA"
	elseif d_ij[q] > d_ik[q]
		mturk[:correct_answers][q] = "optionB"
	elseif d_ij[q] == d_ik[q]
		# mturk[:correct_answers][q] = rand(["optionA", "optionB"])
		mturk[:correct_answers][q] = "optionC"
	end
end

results[:fraction_correct] = 1 - results[:violations]./results[:total]

violations = 1 - dot(results[:total], results[:fraction_correct])/sum(results[:total])
println("Violations for ", task, ": $violations")

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

	d[a,:], μ[a,:] = MTurk.success_function(job, data, bins)

	plot!(d[a,:], μ[a,:],
		xlabel="d_ik - d_ij",
		ylabel="Probability of success", 
		show=true)
end

# filename = string(task, "_annotators", ".mat")
# file = matopen(filename, "w")
# write(file, "d", d)
# write(file, "mu", μ)
# close(file)