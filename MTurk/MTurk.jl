module MTurk

	include("../TripletEmbeddings.jl/src/Embeddings.jl")

	using Random
	using Statistics
	using DataFrames
	using LinearAlgebra

	function hits_per_worker(workers::Array{String,1})
	
		hits = Dict{String,Int}()

		for worker in unique(workers)
			hits[worker] = count(x -> x == worker, workers)
		end

		return hits
	end

	function job_queries(job::DataFrame)

		reference = [parse(Int64, Base.split(job[Symbol("Input.Reference")][i], ".")[1]) for i in 1:size(job,1)]
		optionA = [parse(Int64, Base.split(job[Symbol("Input.A")][i], ".")[1]) for i in 1:size(job,1)]
		optionB = [parse(Int64, Base.split(job[Symbol("Input.B")][i], ".")[1]) for i in 1:size(job,1)]

		return [reference optionA optionB]
	end

	function job_triplets(job::DataFrame)

		queries = job_queries(job)

		triplets = zeros(Int64, size(queries))
		counter = 0

		for i in 1:size(queries,1)
			choice = job[i, Symbol("Answer.choice")]

			if choice  == "optionA"
				# Triplet is in the form (i,j,k)
				counter += 1
				triplets[counter,:] = queries[i,:]'
			elseif choice == "optionB"
				# Correct order for triplet is (i,k,j)
				counter +=1
				triplets[counter, :] = [queries[i,1], queries[i,3], queries[i,2]]'
			end
		end

		return triplets[1:counter,:]

	end

	# function job_violations(job::DataFrame)

	# end

	function job_violations(job::DataFrame, data::Array{Float64,1}; verbose::Bool=false)

		queries = job_queries(job)
		triplets = job_triplets(job)

		violations = 0

		for i in 1:size(triplets,1)
			d_ij = norm(data[triplets[i,1]] - data[triplets[i,2]])
			d_ik = norm(data[triplets[i,1]] - data[triplets[i,3]])
			if d_ij > d_ik && verbose
				@show (d_ij, d_ik)
			end
			violations += d_ij > d_ik
		end

		if verbose
			println("Triplet violations = $violations")
		end
		
		return violations
	end

	function job_quality(job::DataFrame, data::Array{Float64,1}; verbose::Bool=false)

		queries::Array{Int64,2} = job_queries(job)

		optionA::Int64 = 0
		optionB::Int64 = 0
		optionC::Int64 = 0

		totalA::Int64 = 0
		totalB::Int64 = 0
		totalC::Int64 = 0

		for q in 1:size(queries,1)
			choice = job[q, Symbol("Answer.choice")]

			d_ij = norm(data[queries[q,1]] - data[queries[q,2]])
			d_ik = norm(data[queries[q,1]] - data[queries[q,3]])

			if d_ij < d_ik totalA +=1
			elseif d_ij > d_ik totalB +=1
			elseif d_ij == d_ik totalC +=1 end

			if choice  == "optionA"
				# Triplet is in the form (i,j,k)
				optionA += d_ij < d_ik
			elseif choice == "optionB"
				# Correct order for triplet is (i,k,j)
				optionB += d_ij > d_ik
			elseif choice == "optionC"
				# Could not tell the difference
				optionC += d_ij == d_ik
			end
		end



		return optionA, totalA, optionB, totalB, optionC, totalC
	end

	function success_function(job::DataFrame, data::Array{Float64,1}; number_of_bins::Int64 = 10)
		job[Symbol("d_ik - d_ij")] = zeros(size(job,1))
		queries = MTurk.job_queries(job)

		job[:violations] = job[Symbol("Answer.choice")] .== job[:correct_answers]

		for q in 1:size(queries,1)
			i = queries[q,1]
			j = queries[q,2]
			k = queries[q,3]

			if job[:correct_answers][q] == "optionA"
				job[Symbol("d_ik - d_ij")][q] = norm(data[i] - data[k]) - norm(data[i] - data[j])
			elseif job[:correct_answers][q] == "optionB"
				job[Symbol("d_ik - d_ij")][q] = norm(data[i] - data[j]) - norm(data[i] - data[k])
			end
		end

		job = sort(job, Symbol("d_ik - d_ij"))
		bins = range(1, stop=size(job,1), length=number_of_bins)

		μ = zeros(size(bins, 1) - 1) # Probability of success
		distance = zeros(size(bins, 1) - 1) # Mean distance for bin

		for i in eachindex(bins)
			if i < size(bins, 1)
				annotations = job[ceil(Int64, bins[i]):floor(Int64, bins[i] + step(bins)),:]
				μ[i] = mean(annotations[:violations])
				distance[i] = mean(annotations[Symbol("d_ik - d_ij")])
			end
		end

		return distance, μ
	end

	function label_with_answers(data::Array{Float64,1}; probability_success::Array{Float64,3}=ones(size(data,1),size(data,1),size(data,1)))::DataFrame
	    return label_with_answers(reshape(data, size(data,1), 1), probability_success=probability_success)
	end

	function label_with_answers(data::Array{Float64,2}; probability_success::Array{Float64,3}=ones(size(data,1),size(data,1),size(data,1)))::DataFrame
	    # probability represents the probability of swapping the order of a
	    # random triplet

	    # We prealocate the possible total amount of triplets. Before returning,
	    # we clip the array 'triplets' to the amount of nonzero elements.
	    n = size(data,1)
	    triplets = zeros(Int64, n*binomial(n-1, 2), 3)
	    options = zeros(Int64, n*binomial(n-1, 2), 3)
	    correct = Array{String}(undef, n*binomial(n-1, 2))
	    answered = Array{String}(undef, n*binomial(n-1, 2))
	    counter = 0

	    D = Embeddings.distances(data, size(data,1))

	    for k = 1:n, j = 1:k-1, i = 1:n
	        if i != j && i != k

	            @inbounds mistake = probability_success[i,j,k] .<= 1 - rand()

	            if D[i,j] < D[i,k]
	                counter +=1
	                options[counter,:] = [i, j, k]
	                correct[counter] = "optionA"
	                if !mistake
	                    answered[counter] = "optionA"
	                    @inbounds triplets[counter,:] = [i, j, k]
	                else
	                    answered[counter] = "optionB"
	                    @inbounds triplets[counter,:] = [i, k, j]
	                end
	            elseif D[i,j] > D[i,k]
	                counter += 1
	                options[counter,:] = [i, j, k]
	                correct[counter] = "optionB"
	                if !mistake
	                    answered[counter] = "optionB"
	                    @inbounds triplets[counter,:] = [i, k, j]
	                else
	                    answered[counter] = "optionA"
	                    @inbounds triplets[counter,:] = [i, j, k]
	                end
	            end
	        end
	    end

	    return DataFrame([triplets[1:counter,:] options[1:counter,:] correct[1:counter] answered[1:counter]], 
	    	[:i, :j, :k, Symbol("Input.Reference"), Symbol("Input.A"), Symbol("Input.B"), :correct_answers, Symbol("Answer.choice")])
	end

	function split(job::DataFrame, fraction::Real)
	    @assert !isempty(job)
	    
	    amount = floor(Int64, fraction * size(job,1))
	    
	    job = job[shuffle(1:end),:]
	    return job[1:amount, :], job[amount+1:end, :]

	end

end