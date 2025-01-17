require 'torch'
require 'nn'
require 'optim'
require 'cudnn'
require 'LanguageModel'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/tiny-shakespeare.h5')
cmd:option('-input_json', 'data/tiny-shakespeare.json')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-init_from', '')
cmd:option('-resume_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', '1')
cmd:option('-checkpoint_every', '1000')
cmd:option('-checkpoint_name', 'cv/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-cudnn', 0)
cmd:option('-cudnn_fastest',0)

cmd:option('-checkpoint_log',1)

local opt = cmd:parse(arg)

if opt.resume_from ~= '' then
	
	if opt.resume_from:find('_resume') == nil then
		opt.resume_from = opt.resume_from .. '_resume.json'
	end
	if opt.resume_from:find('.json') == nil then
		opt.resume_from = opt.resume_from .. '.json'
	end
	
	local resume = utils.read_json(opt.resume_from)
	opt.init_from = resume.init_from
	opt.reset_iterations = resume.reset_iterations
	opt.input_h5 = resume.input_h5
	opt.input_json = resume.input_json
	opt.batch_size = resume.batch_size
	opt.seq_length = resume.seq_length
	opt.model_type = resume.model_type
	opt.wordvec_size = resume.wordvec_size
	opt.rnn_size = resume.rnn_size
	opt.num_layers = resume.num_layers
	opt.dropout = resume.dropout
	opt.batchnorm = resume.batchnorm
	opt.max_epochs = resume.max_epochs
	opt.learning_rate = resume.learning_rate
	opt.grad_clip = resume.grad_clip
	opt.lr_decay_every = resume.lr_decay_every
	opt.lr_decay_factor = resume.lr_decay_factor
	opt.print_every = resume.print_every
	opt.checkpoint_every = resume.checkpoint_every
	opt.checkpoint_name = resume.checkpoint_name
	opt.speed_benchmark = resume.speed_benchmark
	opt.memory_benchmark = resume.memory_benchmark
	opt.gpu = resume.gpu
	opt.gpu_backend = resume.gpu_backend
	opt.cudnn = resume.cudnn
	opt.cudnn_fastest = resume.cudnn_fastest
	if resume.checkpoint_log ~= nil then
		opt.checkpoint_log = resume.checkpoint_log
	end
end

local epoch_checkpoint = false
local checkpoint_every = 1000
local epoch_print = false
local print_every = 1

if opt.checkpoint_every:sub(-1,-1):upper() == 'E' then
	epoch_checkpoint = true
	checkpoint_every = tonumber(opt.checkpoint_every:sub(1,-2))
else
	checkpoint_every = tonumber(opt.checkpoint_every)
end

if opt.print_every:sub(-1,-1):upper() == 'E' then
	epoch_print = true
	print_every = tonumber(opt.print_every:sub(1,-2))
else
	print_every = tonumber(opt.print_every)
end

-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end

-- Initialize the model and criterion
local opt_clone = torch.deserialize(torch.serialize(opt))
opt_clone.idx_to_token = idx_to_token
local model = nil
local start_i = 0
if opt.init_from ~= '' then
	if opt.init_from:find('.t7') == nil then
		opt.init_from = opt.init_from .. '.t7'
	end
  print('Initializing from ', opt.init_from)
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  if opt.reset_iterations == 0 then
    start_i = checkpoint.i
  end
else
  model = nn.LanguageModel(opt_clone):type(dtype)
end
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    print(string.format('Using %dMB of memory', memory_used_mb))
    table.insert(memory_usage, memory_used)
  end

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end

-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
model:training()
local train_losses = 0
local train_count = 0

-- these are only used if printing or checkpointing every epoch
local last_print = math.floor((start_i/num_train)/print_every)
local last_check = math.floor((start_i/num_train)/checkpoint_every)

for i = start_i + 1, num_iterations do
  local epoch = math.floor(i / num_train) + 1

  -- Check if we are at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states

    -- Maybe decay learning rate
    if epoch % opt.lr_decay_every == 0 then
      local old_lr = optim_config.learningRate
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
    end
  end
  
  local float_epoch = i / num_train

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params, optim_config)
  table.insert(train_loss_history, loss[1])
  train_losses = train_losses + loss[1]
  train_count = train_count + 1
  local do_print = false
  if print_every > 0 then
    if epoch_print then
		if float_epoch/print_every - last_print >= 1.0 then
			do_print = true
			last_print = last_print + 1
		end
    else
      do_print = (i % print_every == 0)
    end
  end
  
  if do_print then
    
    local msg = 'Epoch %.2f / %d, i = %d / %d, mean loss = %f'
	local mean_loss = train_losses / train_count
	train_losses = 0
	train_count = 0
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, mean_loss}
    print(string.format(unpack(args)))
  end

  -- Maybe save a checkpoint
  local check_every = checkpoint_every
  local do_checkpoint = false
  if check_every > 0 then
    if epoch_checkpoint then
		if float_epoch/check_every - last_check >= 1.0 then
			do_checkpoint = true
			last_check = last_check + 1
		end
    else
      do_checkpoint = (i % check_every == 0)
    end
  end
	
  
  if do_checkpoint or i == num_iterations then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
	
	local checkpoint_number
	if epoch_checkpoint then
		checkpoint_number = string.format('%ge',checkpoint_every * last_check)
	else
		checkpoint_number = string.format('%d',i)
	end
	
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      local N_v = xv:size(1)
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N_v * T)
      local scores = model:forward(xv):view(N_v * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
	print(string.format('Saving Checkpoint %s_%s',opt.checkpoint_name, checkpoint_number))
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()
	
    if opt.checkpoint_log ~= 0 then
      -- First save a JSON checkpoint, excluding the model
      local log_checkpoint = {
        train_loss_history = train_loss_history,
        val_loss_history = val_loss_history,
        val_loss_history_it = val_loss_history_it,
        forward_backward_times = forward_backward_times,
        memory_usage = memory_usage,
        i = i
      }
      local filename = string.format('%s_%s_log.json', opt.checkpoint_name, checkpoint_number)
      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, log_checkpoint)
    end
	
	local resume_filename = string.format('%s_%s.t7', opt.checkpoint_name, checkpoint_number)
	-- Save a resume point with all options needed to restart training
	local resume_checkpoint = {
		init_from = resume_filename,
		reset_iterations = 0,
		input_h5 = opt.input_h5,
		input_json = opt.input_json,
		batch_size = opt.batch_size,
		seq_length = opt.seq_length,
		model_type = opt.model_type,
		wordvec_size = opt.wordvec_size,
		rnn_size = opt.rnn_size,
		num_layers = opt.num_layers,
		dropout = opt.dropout,
		batchnorm = opt.batchnorm,
		max_epochs = opt.max_epochs,
		learning_rate = optim_config.learningRate,
		initial_learning_rate = opt.learning_rate,
		grad_clip = opt.grad_clip,
		lr_decay_every = opt.lr_decay_every,
		lr_decay_factor = opt.lr_decay_factor,
		print_every = opt.print_every,
		checkpoint_every = opt.checkpoint_every,
		checkpoint_name = opt.checkpoint_name,
		speed_benchmark = opt.speed_benchmark,
		memory_benchmark = opt.memory_benchmark,
		gpu = opt.gpu,
		gpu_backend = opt.gpu_backend,
		cudnn = opt.cudnn,
		cudnn_fastest = opt.cudnn_fastest,
		checkpoint_log = opt.checkpoint_log
	}
	local filename = string.format('%s_%s_resume.json', opt.checkpoint_name, checkpoint_number)
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, resume_checkpoint)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    model:float()
    local model_checkpoint = {
	  model = model,
      i = i
    }
    filename = string.format('%s_%s.t7', opt.checkpoint_name, checkpoint_number)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, model_checkpoint)
    model:type(dtype)
    params, grad_params = model:getParameters()
    collectgarbage()
  end
end
