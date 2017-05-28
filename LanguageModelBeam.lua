require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTMBeam'

local utils = require 'util.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true -- RS EDIT: WATCH THIS
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function LM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function LM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  self.net:training()
  parent.training(self)
end


function LM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function LM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


-- RS EDIT
function LM:pushStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:pushStates()
  end
end


function LM:popStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:popStates()
  end
end
-- /RS EDIT


function LM:encode_string(s)
  local encoded = {}
  local i = 1
  while i <= #s do
    local token = s:sub(i, i)
    local idx = self.token_to_idx[token]
    i = i + 1
    if not idx then
      token = s:sub(i-1, i)
      idx = self.token_to_idx[token]
      i = i + 1
    end
    assert(idx ~= nil, 'Got invalid idx')
    encoded[#encoded+1] = idx
  end

  return torch.LongTensor(encoded)
end


function LM:decode_string(encoded)
  assert(torch.isTensor(encoded))
  assert(encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
RS edit

Sample from the language model until it reaches a terminator character.

Inputs:
- start_text: string, can be ""
- terminator_chars: class of chars in Lua match format, e.g. "[!?\\.]" (note the double escape, ugh)
- min_num_words: if terminator char reached before this threshold, keep going until the next one

Note temperature table; probably worth fiddling with.

Returns:
- the generated string!
--]]


function LM:sample(start_text, terminator_chars, min_num_words, max_chars)
  self:resetStates()
  local scores

  if #start_text > 0 then
    -- warm up model with start text (but don't add to sampled string)
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2) -- length of start_text
    scores = self:forward(x)[{{}, {T0, T0}}]
  else
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
  end

  local terminated = false
  local num_words_approx = 1

  local temps = {0.5, 0.6, 0.7, 0.8, 0.9}
  local temp = temps[math.random(#temps)] -- for this run

  local next_char_idx = nil
  local next_char = nil
  local sampled_string = ''

  local max_length_to_generate = max_chars

  while (not terminated) and (#sampled_string < max_length_to_generate) do

    local probs = torch.div(scores, temp):double():exp():squeeze()
    probs:div(torch.sum(probs))
    next_char_idx = torch.multinomial(probs, 1):view(1, 1)
    scores = self:forward(next_char_idx)

    next_char = self.idx_to_token[next_char_idx[1][1]]

    sampled_string = sampled_string .. next_char

    if next_char == ' ' then
      num_words_approx = num_words_approx + 1 -- close enough
    end

    if next_char:match(terminator_chars) then
      if num_words_approx > min_num_words then
        terminated = true
      end
    end

  end

  self:resetStates()
  return sampled_string
end


--[[
RS edit

Sample from the language model until it reaches a terminator character.

Inputs:
- start_text: string, can be ""
- terminator_chars: class of chars in Lua match format, e.g. "[!?\\.]" (note the double escape, ugh)
- min_num_words: if terminator char reached before this threshold, keep going until the next one
- max_chars: max # of chars to generate overall, just in case
- search_width: how many potential chars to look at each step
- search_depth: how far to explore down those paths

Returns:
- the generated string!
--]]


function LM:sample_search(start_text, terminator_chars, min_num_words, max_chars, search_width, search_depth)
  self:resetStates()

  local working_sequence = self:encode_string(start_text):view(1, -1) -- boot up the sequence
  local T0 = working_sequence:size(2) -- length of this sequence

  -- warm up the network with this sequence
  local working_scores = self:forward(working_sequence)[{{}, {T0, T0}}]

  local temps = {0.7, 0.8, 0.9, 1.0}
  math.randomseed(os.time())
  temp = 1.8 --temps[math.random(#temps)] -- for this run

  local running = true
  local num_words_approx = 0

  while running do

    -- based on current state, we get the probabilities for the next char...
    local char_probs = torch.div(working_scores, temp):exp():squeeze()
    char_probs:div(torch.sum(char_probs))

    -- now we sample the top n=search_width char candidates
    local sample = torch.multinomial(char_probs, search_width, false)

    -- and record their corresponding probs, which we will be adding to in a moment
    local cumulative_char_probs = torch.FloatTensor(search_width)
    for i=1,search_width do
      cumulative_char_probs[i] = char_probs[sample[i]]
    end

    -- now, we explore those sampled candidate chars
    for sample_index=1,search_width do
      -- we are going to experiment, so let's save the state of the model first...
      self:pushStates()

      -- this is the char whose path we are exploring
      local candidate_char_idx = sample[sample_index]

      -- search_depth: how far do we want to travel down each one of these paths?
      for depth=1,search_depth do
        -- first, tell the model: we are choosing this char (hypothetically)
        local scores = self:forward(torch.LongTensor({{candidate_char_idx}})) -- ugly

        -- now, ask the model: what would the probs for the NEXT char look like?
        local next_char_probs = torch.div(scores, temp):exp():squeeze()
        next_char_probs:div(torch.sum(next_char_probs))

        -- NOTE: CHECK THE TOPK FUNCTION HERE >:(
        -- OH I GUESS IT'S OKAY... it knows what that true is all about
        -- here we just look at the MOST likely next char if we follow this path...
        local most_likely_char_prob, most_likely_char_idx = torch.topk(next_char_probs, 1, true) -- true is important

        -- tally the joint probability of this path
        cumulative_char_probs[sample_index] = cumulative_char_probs[sample_index] * most_likely_char_prob[1]

        -- if looping back around to push forward down this path, we'll need this
        candidate_char_idx = most_likely_char_idx[1]
      end

      -- we're done with our experiment, THANKS
      self:popStates()
    end

    -- now that we've summed up the joint probabilities of these different paths
    -- to a reasonable depth, we can find the "best" one
    local _, sorted_char_indices = torch.sort(cumulative_char_probs, 1, true) -- false would be: sort in ascending order

    -- HERE'S THE MOMENT OF TRUTH
    next_char_idx = sample[sorted_char_indices[1]]

    -- let's tell the model about that immediately so we don't forget
    working_scores = self:forward(torch.LongTensor({{next_char_idx}})) -- likewise: UGLY

    -- and add the char to the working sequence
    working_sequence = appendCharToSequence(next_char_idx, working_sequence)

    -- we need to get that char index as a real char to match it against terminator chars...
    local next_char = self.idx_to_token[next_char_idx]

    if next_char == ' ' then
      num_words_approx = num_words_approx + 1 -- close enough
    end

    if (working_sequence:size(2) > max_chars + T0) then -- or next_char:match(terminator_chars)
      -- if num_words_approx >= min_num_words then
        -- we're finished
        running = false
      -- end
    end

    if working_sequence:size(2) % 100 == 0 then
      print('Total symbols: ' .. working_sequence:size(2))
    end

  end -- end while loop

  finished_string = ""
  -- translate from char indexes to actual characters and build the string
  -- begin at T0+1 because we don't need to take any start_text with us
  for char_index=T0+1,working_sequence:size(2) do -- once again, size(2) is the seq length
    finished_string = finished_string .. self.idx_to_token[working_sequence[1][char_index]] -- the [1] is hella tricky
  end

  self:resetStates()
  return finished_string
end


-- RS: this probably shouldn't be global...
function appendCharToSequence(char_idx, base_sequence)
  local new_size = torch.LongStorage({1, base_sequence:size(2)+1}) -- size(2) is # of chars
  local extended_sequence = base_sequence:resize(new_size) -- make it bigger
  extended_sequence[1][-1] = char_idx -- add the char to the end
  return extended_sequence
end


function LM:clearState()
  self.net:clearState()
end
