include("../Embeddings/src/utilities.jl")

using DataFrames
using CSV

srand(4)

task = "TaskA.csv"
first_annotation = 1001
last_annotation = 47052

data = load_data(path=joinpath("data/", task))

n = size(data,1)

triplets = unique_triplets(n)

url = joinpath("https://s3-us-west-1.amazonaws.com/mturk-web-hosting/TaskA_frames/")

df = DataFrame(url = fill(url, size(triplets,1)),
					 Reference = string.(triplets[:,1], ".png"),
					 A = string.(triplets[:,2], ".png"),
					 B = string.(triplets[:,3], ".png"))

df = df[shuffle(1:end),:]

df = df[first_annotation:last_annotation,:]

CSV.write(string("Mturk/input_", split(task, ".")[1], "_", first_annotation, "_", last_annotation, ".csv"), df)