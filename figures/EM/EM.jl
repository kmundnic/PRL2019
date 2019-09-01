using Glob
using TableReader
using StatsPlots

task = "TaskA"
files = glob(string(task, "*.csv"), expanduser("~/Documents/Research/PRL2019/results/EM/"))
data = [readcsv(file, hasheader=false, colnames=[split(basename(file), '.')[1]]) for file in files]
data = hcat(data...)
data[!, :time] = 1:size(data,1)

columns = permutedims(sort(Symbol.(getindex.(split.(basename.(files), '.'), 1))))

@df data plot(:time, cols(columns))