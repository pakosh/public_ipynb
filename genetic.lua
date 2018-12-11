-- require 'torch'

-- dqn = {}
local threads = require 'threads'

local gen = torch.class('dqn.genetic')

--file = io.write( 'run.log', "w+")

local extractedNet = nil
local populationSize = nil

local function getNetLength( agent, extract, net_length )
	
	local extractedNet = agent.network:getParameters()
	local hromosomeTensorOffset = extractedNet:size()[1]
	-- print( extractedNet, extractedNet:size() )

	print( 'total net length: ', hromosomeTensorOffset )
	return hromosomeTensorOffset, extractedNet
end

local function get_by_probability( probabilities )
	local i
	local probval = torch.rand(1)[1]
	for i=1, probabilities:size()[1] do
		probval = probval - probabilities[i]
		-- print( i, probval )
		if probval <= 0 then 
			return i
		end
	end
end

local function choose_parents( probabilities )
	local  a, b
	a = get_by_probability( probabilities ) 
	b = get_by_probability( probabilities )
	return a, b
end

local crossoverPoints = 64
local function crossover(a, b, crossoverRate, gpu)
	if torch.uniform() > crossoverRate then
		return a
	end
	
	local s = torch.Tensor( a:size() )
	s:copy( a )

	local cut = {}
	cut[1] = a:size()[1]

	for i=2, crossoverPoints+1 do
		cut[i] = torch.random( 1, cut[1] )
	end
	cut[#cut + 1] = 1

	table.sort(cut)
	
	for i=1, #cut, 2 do

		s[ {{ cut[i], cut[i+1] }, 1 } ]:copy( b[ {{ cut[i], cut[i+1] }, 1 } ] )
	end

	return s
end

local function mutation(bitstring, mutationRate)

	local rnd = torch.rand(bitstring:size()[1], 1)
	bitstring = bitstring - (rnd - 0.5) * mutationRate
	bitstring[ bitstring:gt(1) ] = bitstring[ bitstring:gt(1) ] - 2
	return bitstring
end

local mutationRate = nil
local crossoverRate = nil
local calc_threads = nil
local ngames = nil

function gen:__init(args)
	-- configuration
	print('init: ', args)
	self.mutationRate = args.mutationRate or 0.01
	self.crossoverRate = args.crossoverRate or 0.99
	self.populationSize = args.populationSize or 20
	populationSize = self.populationSize

	self.maxGenerations = args.maxGenerations or 10000
	self.selectionTournamentSize = args.selectionTournamentSize or 3

	self.threads = args.eval_threads or 2
	self.gpu = args.gpu or -1
	self.best_hromosome_file = args.best_hromosome_file or nil
	self.ngames = args.ngames or 2
	self.stagnation = args.stagnation
	self.liderMutations = args.liderMutations or 1

	ngames = self.ngames
	calc_threads = self.threads
	mutationRate = self.mutationRate
	crossoverRate = self.crossoverRate

	print('genetic args done', self.eval_steps, self.agent, self.game_env, self.hromosomeSize)
	print('populationSize', self.populationSize)
	print('setup funct: ', setup)

	local g, ga, a, o = setup(args)

	self.loadedNet = a.loadedNet or false

	print( '!!loadedNet ? ', a.loadedNet )

	self.net_length, extractedNet = getNetLength( a )
	self.hromosomeSize = self.net_length

	local net_length = self.net_length

	g = nil
	ga = nil
	a = nil
	o = nil

	collectgarbage()

	self.pool = threads.Threads(
		self.threads,
		function(threadid)
			print('starting a new thread/state number ' .. threadid)
			if not dqn then
			    require "initenv"
			end
			opt = args
			gpu = args.gpu or -1
			game_env, game_actions, agent, opt = setup(opt)

			opt = nil

			hromo = agent.network:getParameters()
			agent.threadid = threadid

			collectgarbage()			
		end
	)

	collectgarbage()
end

function gen:initPopulation()

	print(self.populationSize, self.net_length)

    local t = torch.rand( self.populationSize, self.net_length, 1)

    local net = nil

	if not self.loadedNet and (type(self.best_hromosome_file) == 'string' and self.best_hromosome_file ~= "") then
		print(self.best_hromosome_file)
    	local msg, net = pcall( torch.load, self.best_hromosome_file )
	    if not msg or not net then
	        error("Error loading preprocessing net")
	    else
	    	print('net loaded ', self.best_hromosome_file)
	    	
	    end
    elseif self.loadedNet then
    	print('extractPretrained')    	
    	net = extractedNet
    end

    if net then
	    for i=1, self.populationSize do
			t[{i}]:copy(net)
		end
	end

	return t
end


local function fitness(agent, game_env)
	local gameActions = game_env:getActions()
	local screen, reward, terminal = game_env:newGame()
    local nrewards = 0
    local nepisodes = 1
    local episode_reward = 0

	local eval_time = sys.clock()
	local eval_steps = agent.eval_steps

	local esteps = 0

    for estep=1, eval_steps do
        local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

        screen, reward, terminal = game_env:step( gameActions[action_index] )
		-- print( '::: screen ::: ', screen )
		-- print( '::: reward ::: ', reward )
		-- print( '::: terminal ::: ', terminal )
        if estep%10000 == 0 then 
        	print(string.format(
			            'Episode_reward: %.10f (nrewards: %d), testing time: %ds, ' ..
			            'testing rate: %dfps, esteps: %d, nepisodes: %d, threadid: %d',
			            (episode_reward/nepisodes), (nrewards/nepisodes), (sys.clock() - eval_time), (agent.actrep*estep/(sys.clock() - eval_time)), estep, nepisodes, agent.threadid  ))
        	collectgarbage() 
        end

        -- record every reward
        esteps = esteps + 1

        -- print( reward ~= 0 )
        if reward ~= 0 then
        	episode_reward = episode_reward + reward
           	nrewards = nrewards + 1
        end

        if terminal then
            if nepisodes >= ngames then
            	if nrewards/nepisodes >1 then
            		episode_reward = episode_reward + episode_reward/nrewards/nrewards
            	end
            	episode_reward = episode_reward + esteps/eval_steps
            	break
            end
            nepisodes = nepisodes + 1
			screen, reward, terminal = game_env:nextRandomGame()
            -- break
        end
    end

    episode_reward = episode_reward/nepisodes
    -- nrewards = nrewards/nepisodes

    eval_time = sys.clock() - eval_time
    
	collectgarbage()
    return episode_reward, nrewards/nepisodes, eval_time, esteps

end

function gen:reproduce(population, fitnesses)
	-- calc probabilities
	-- print('fitnesses :: ', fitnesses)
	local reverseSum = fitnesses:sum(1)[{1,1}]
	-- print( 'reverseSum :: ', reverseSum )
	local probabilities = fitnesses:select(2,1)
	-- print( 'probabilities :: ', probabilities )
	probabilities:div(reverseSum)

	-- print( probabilities )
	print('generate new population')
	local cnt = 3
	
	-- local pop = nil
	-- if self.gpu >= 0 then
	-- 	pop = torch.CudaTensor( self.populationSize, self.net_length, 1)
	-- else
	local pop = torch.Tensor( self.populationSize, self.net_length, 1)
	-- end

	local population = population
	-- local threads = self.threads

	if cnt < self.populationSize then
		repeat

			self.pool:addjob(
				function()
					-- print(string.format('active -- thread ID is %x', __threadid))
					-- print('crossoverRate/mutationRate', crossoverRate, mutationRate)
					local p1, p2 = choose_parents( probabilities )
					collectgarbage()
					-- print('p1 p2', p1, p2, cnt)
					local child = crossover( population[p1], population[p2], crossoverRate )
					collectgarbage()
					-- print('child', child)
					if torch.uniform() <= mutationRate then
						child = mutation( child, mutationRate )						
						collectgarbage()
					end

					return child, __threadid, p1, p2, cnt
				end,

				function(child, id, p1, p2, cnt)
					print('p1 p2', p1, p2, cnt)
					-- print('score, id :: ', score, id)
					pop[cnt]:copy( child )					
					
					-- if cnt%calc_threads == 0 then 
			        	collectgarbage() 
			        -- end
				end
			)
			
			cnt = cnt + 1
			
		until cnt == self.populationSize
		self.pool:synchronize()
	end
	
	return pop
end

function gen:evolve()

	print('start evolve')
	local population = self:initPopulation()

	print(' population inited :: ', population:size())
	local bestString = nil

	local bestScore = 0
	local bestGen = nil
	local score = 0
	local stagnation = 0
	
	local fitnesses = torch.Tensor( self.populationSize, 1 )

	for generation=1, self.maxGenerations do
		print( 'generation :: ', generation, 'lider mutation: ', self.liderMutations )

		local jobdone = 0
		-- local threads = self.threads
		for i=1, self.populationSize do

			local v = population[ i ]
			-- print( v )

			self.pool:addjob(
				function()			
					local all_eval_time = sys.clock()	
					hromo:copy( v )
					local score, nrewards, eval_time, esteps = fitness( agent, game_env)
			        all_eval_time = sys.clock() - all_eval_time
					return score, __threadid, nrewards, eval_time, agent.actrep, esteps, all_eval_time, agent
				end,

				function(score, id, nrewards, eval_time, actrep, esteps, all_eval_time, agent)
					-- population[i]:eq( v ):sum()
					print(string.format(
			            '\nEpisode_reward: %.10f (nrewards: %d), testing time: %ds, ' ..
			            'testing rate: %dfps, esteps: %d, alletime: %ds, crossoverRate/mutationRate: %.2f/%.2f',
			            score, nrewards, eval_time, actrep*esteps/eval_time, esteps/ngames, all_eval_time, crossoverRate, mutationRate  ))
					-- print('score, id :: ', score, id)
					fitnesses[{i, 1}] = score
					print(string.format("task %d finished (ran on thread ID %x, score: %.10f, bestScore was %.10f, generation: %d)", i, id, score, bestScore, generation) )
					jobdone = jobdone + 1

					if(score > bestScore) then
						bestScore = score
						bestGen = population[i]:clone()
						stagnation = 0
						crossoverRate = 0.99
						mutationRate = 0.01
						torch.save( "bestnet_" .. score .. ".t7", bestGen)
					end

					if i%calc_threads == 0 then 
			        	collectgarbage() 
			        end
				end
			)
		end

		collectgarbage()

		self.pool:synchronize()
		collectgarbage()

		stagnation = stagnation + 1

		if stagnation > self.stagnation then
			print( '!! stagnation ')
			if mutationRate < 1 then
				mutationRate = mutationRate + 0.2
				if mutationRate > 1 then
					mutationRate = 1
				end
			end
			if crossoverRate >= 0.2 then
				crossoverRate = crossoverRate - 0.2
				if crossoverRate <= 0 then
					crossoverRate = 0.2
				end
			end
			if crossoverRate <= 0.2 then
				self.liderMutations = self.liderMutations + 1
				if self.liderMutations >= self.populationSize then
					self.liderMutations = self.populationSize - 1
				end
			end
		end
		print( 'generation :: ', generation, ' done. Best score :: ', bestScore )
--		file:write("\n" .. string.format('generation :: %d done. Best score :: %.10f', i, score) );

		-- -- -- select and reproduce
		if self.liderMutations - 1 ~= self.populationSize then 
			local pop = self:reproduce(population, fitnesses)
			population:copy(pop)

		end

		population[ 1 ] = bestGen
		for z=2, self.liderMutations + 1 do
			population[ z ] = mutation( bestGen:clone(), mutationRate )
		end
		collectgarbage()
	end	
	return bestGen
end