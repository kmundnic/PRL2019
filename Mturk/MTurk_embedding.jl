include("../TripletEmbeddings.jl/src/Embeddings.jl")
include("MTurk.jl")

using MAT
using CSV
using Glob
using Plots; gr()
using Printf
using Interpolations

task = "TaskA"
println("Computing embedding for ", task, " from MTurk annotations")

files = glob(string(task, "_output*.csv"))
mturk = vcat(CSV.read.(files)...)

queries = MTurk.job_queries(mturk)

triplets = MTurk.job_triplets(mturk)

dimensions = 1
params = Dict{Symbol,Real}()
params[:α] = 2.0
params[:σ] = 1\sqrt(2)
params[:μ] = 20

no_triplets = floor(Int64, size(triplets,1)/4)
# te = Embeddings.tSTE(triplets[1:no_triplets,:], dimensions, params)
# te = Embeddings.STE(triplets[1:no_triplets,:], dimensions, params)
# te = Embeddings.HingeGNMDS(triplets[1:no_triplets,:], dimensions)
te = Embeddings.CKL(triplets[1:no_triplets,:], dimensions, params)

@time violations = Embeddings.compute(te; max_iter=1000)

data = Embeddings.load_data(path=string("../data/", task, ".csv"))

Y, mse = Embeddings.scale(data, dropdims(Embeddings.X(te), dims=2), MSE=true)

# We need to shift 0.5[s] due to how ffmpeg samples the video
xs = 1:size(Embeddings.X(te), 1)
A = [Y[x] for x in xs]
interp_linear = LinearInterpolation(xs, A)

Z = copy(Y)

for x in xs
	if 2 <= x <= size(Y,1)
		Z[x] = interp_linear(x - 0.5)
	end
end

_, mse = Embeddings.scale(data, Z, MSE=true)

println("MSE = $mse")
@printf("Violations = %.2f %%\n", 100*violations)

plot(data, label="Data", color=:black)
plot!(Z, label="MTurk", color=:blue)
plot!(1.5:1:(size(Y,1)+1), Y, label="MTurk shifted", color=:green)


# filename = string(task, "_", string(no_triplets),  "_.mat")
# file = matopen(filename, "w")
# write(file, "data", data)
# write(file, "Y", Y)
# write(file, "mse", mse)
# write(file, "violations", violations)
# close(file)