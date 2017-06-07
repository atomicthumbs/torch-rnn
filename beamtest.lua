require 'torch'
require 'nn'
require 'LanguageModelBeam'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'checkpoints/scifi-model.t7')
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)

local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  print('Running in CPU mode... it will be slow!')
end

model:evaluate()

local processed_start_text = '=====\nАвтор: Бутырка\nНазвание: Воровские законы'
target_num = 1

local t0 = os.clock()
local generated = {}

while #generated < target_num do
  -- start_text: string, can be ""
  -- terminator_chars: class of chars in Lua match format, e.g. "[!?\\.]" (note the double escape, ugh)
  -- min_num_words: if terminator char reached before this threshold, keep going until the next one
  -- max_chars: max # of chars to generate overall
  -- search_width: how many potential chars to look at each step
  -- search_depth: how far to explore down those paths
  
  -- sample_search(start_text, terminator_chars, min_num_words, max_chars, search_width, search_depth)

  local sentence = model:sample_search(processed_start_text, '[]', 999, 1400, 4, 7)

  print("sentence:")
  print(sentence)
  table.insert(generated, sentence)
end

local elapsed = string.format('%.2f', os.clock() - t0)

print('Took ' .. elapsed .. ' of some undetermined unit of time')