--[[
 This paper is from the offcial implementation on torch
This file implements Batch Normalization as described in the paper:
   "Batch Normalization: Accelerating Deep Network Training
                         by Reducing Internal Covariate Shift"
                   by Sergey Ioffe, Christian Szegedy
   This implementation is useful for inputs NOT coming from convolution layers.
   For Convolution layers, see SpatialBatchNormalization.lua
   The operation implemented is:
   y =     ( x - mean(x) )
        -------------------- * gamma + beta
       standard-deviation(x)
   where gamma and beta are learnable parameters.
   The learning of gamma and beta is optional.
   Usage:
   with    learnable parameters: nn.BatchNormalization(N [, eps] [,momentum])
                                 where N = dimensionality of input
   without learnable parameters: nn.BatchNormalization(0 [, eps] [,momentum])
   eps is a small value added to the standard-deviation to avoid divide-by-zero.
       Defaults to 1e-5
   In training time, this layer keeps a running estimate of it's computed mean and std.
   The running sum is kept with a default momentup of 0.1 (unless over-ridden)
   In test time, this running mean/std is used to normalize.
]]--
local BatchLinear_FIM,parent = torch.class('nn.BatchLinear_FIM', 'nn.Module')

function BatchLinear_FIM:__init(nOutput, affine, eps, momentum)
   parent.__init(self)
   assert(nOutput and type(nOutput) == 'number',
          'Missing argument #1: dimensionality of input. ')
   assert(nOutput ~= 0, 'To set affine=false call BatchNormalization'
     .. '(nOutput,  eps, momentum, false) ')
   if affine ~= nil then
      assert(type(affine) == 'boolean', 'affine has to be true/false')
      self.affine = affine
   else
      self.affine = false
   end
   self.eps = eps or 1e-5
   self.train = true
   self.momentum = momentum or 0.1
   self.running_mean = torch.zeros(nOutput)
   self.running_std = torch.ones(nOutput)

   if self.affine then
      self.weight = torch.Tensor(nOutput)
      self.bias = torch.Tensor(nOutput)
      self.gradWeight = torch.Tensor(nOutput)
      self.gradBias = torch.Tensor(nOutput)
      self:reset()
   end
   
   self.isCalculateFIM=true
   
   
   --for debug
   self.debug=false
   self.debug_detailInfo=false
   self.printInterval=1
   self.count=0
end

function BatchLinear_FIM:reset()
   self.weight:uniform()
  -- self.weight:fill(1)
   self.bias:zero()
   self.running_mean:zero()
   self.running_std:fill(1)
end

function BatchLinear_FIM:updateOutput(input)
   assert(input:dim() == 2, 'only mini-batch supported (2D tensor), got '
             .. input:dim() .. 'D tensor instead')
   local nBatch = input:size(1)

   -- buffers that are reused
   self.buffer = self.buffer or input.new()
   self.buffer2 = self.buffer2 or input.new()
   self.centered = self.centered or input.new()
   self.centered:resizeAs(input)
   self.std = self.std or input.new()
   self.normalized = self.normalized or input.new()
   self.normalized:resizeAs(input)
   self.output:resizeAs(input)
   self.gradInput:resizeAs(input)
   if self.train == false then
     if self.debug then 
      print('--------------------------batch:test mode-------------------')
     end
      self.output:copy(input)
      self.buffer:repeatTensor(self.running_mean, nBatch, 1)
      self.output:add(-1, self.buffer)
      self.buffer:repeatTensor(self.running_std, nBatch, 1)
     -- print(self.running_std:clone():pow(-1):mean())
      self.output:cmul(self.buffer)
   else -- training mode
      -- calculate mean over mini-batch
      if self.debug  then  
       print('--------------------------batch:train mode-------------------')
      end
      self.buffer:mean(input, 1)                        -- E(x) = expectation of x.
      self.running_mean:mul(1 - self.momentum):add(self.momentum, self.buffer) -- add to running mean
      self.buffer:repeatTensor(self.buffer, nBatch, 1)

      -- subtract mean
      self.centered:add(input, -1, self.buffer)         -- x - E(x)

      -- calculate standard deviation over mini-batch
      self.buffer:copy(self.centered):cmul(self.buffer) -- [x - E(x)]^2

      -- 1 / E([x - E(x)]^2)
      self.std:mean(self.buffer, 1):add(self.eps):sqrt():pow(-1)
      self.running_std:mul(1 - self.momentum):add(self.momentum, self.std) -- add to running stdv
      self.buffer:repeatTensor(self.std, nBatch, 1)
      
      if self.debug and (self.count % self.printInterval==0) then
        print('--------the scale value-------------')
        print(self.std)
      end
      
      -- divide standard-deviation + eps
      self.output:cmul(self.centered, self.buffer)
      self.normalized:copy(self.output)
   end

   if self.affine then
      -- multiply with gamma and add beta
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.output:cmul(self.buffer)
      self.buffer:repeatTensor(self.bias, nBatch, 1)
      self.output:add(self.buffer)
   end
   
    if self.debug  then
     
       self.buffer:resize(self.output:size(2),self.output:size(2))
       self.buffer:addmm(0,self.buffer,1/input:size(1),self.output:t(),self.output) ---the validate matrix    
    --   print("------debug_batch_module:diagonal of validate matrix------")
     --  print(self.buffer)
       
      local rotation,eig,_=torch.svd(self.buffer)
      print("-------debug_eig of the correlate matrix r.w.t nomalized activation-----")
      print(eig)
      
       print("-------debug_dignoal of the correlate matrix r.w.t nomalized activation-----")
       for i=1,self.buffer:size(1) do
          print(i..': '..self.buffer[i][i])
       end
  
  
    end
    
    if self.debug_detailInfo and (self.count % self.printInterval==0)then

 
      local input_mean=input:mean(1)
      local input_normPerDim=torch.norm(input,1,1)/input:size(1)
      
      local output_mean=self.output:mean(1)
      local output_normPerDim=torch.norm(self.output,1,1)/self.output:size(1)
     
      print('debug_batchModule--input_mean:') 
      print(input_mean)
      print('debug_batchModule--input_normPerDim:') 
      print(input_normPerDim)    
      print('debug_batchModule--output_mean:') 
      print(output_mean)    
      print('debug_batchModule--output_normPerDim:') 
      print(output_normPerDim)        
   end
  --  print('-----------BN:output:--------')
  --  print(self.output)
   return self.output
end

function BatchLinear_FIM:updateGradInput(input, gradOutput)
   assert(input:dim() == 2, 'only mini-batch supported')
   assert(gradOutput:dim() == 2, 'only mini-batch supported')
 --  assert(self.train == true, 'should be in training mode when self.train is true')
   local nBatch = input:size(1)

 if self.train==false then
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.buffer:repeatTensor(self.running_std, nBatch, 1)
   self.gradInput:cmul(self.buffer)
 else
   self.gradInput:cmul(self.centered, gradOutput)
   self.buffer:mean(self.gradInput, 1)
   self.gradInput:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:cmul(self.centered):mul(-1)
   self.buffer:repeatTensor(self.std, nBatch, 1)
   self.gradInput:cmul(self.buffer):cmul(self.buffer)

   self.buffer:mean(gradOutput, 1)
   self.buffer:repeatTensor(self.buffer, nBatch, 1)
   self.gradInput:add(gradOutput):add(-1, self.buffer)
   if self.debug_detailInfo then
     print('-----------------hidden gradInput:-----------')
     print(self.gradInput[{{1,20},{}}]) 
   end
   
   self.buffer:repeatTensor(self.std, nBatch, 1)
   self.gradInput:cmul(self.buffer)
  end
   
   
   
   if self.affine then
      self.buffer:repeatTensor(self.weight, nBatch, 1)
      self.gradInput:cmul(self.buffer)
   end

   -------------debug information------------
    if self.debug_detailInfo and (self.count % self.printInterval==0)then
      local gradOutput_norm=torch.norm(gradOutput,1)
      local gradInput_norm=torch.norm(self.gradInput,1)
     
      print('debug_batchModule--gradOutput_norm_elementWise:'..gradOutput_norm..' --gradInput_norm_elementWise:'..gradInput_norm)
      
      
    end

    if self.debug_detailInfo and (self.count % self.printInterval==0)then

      local gradInput_mean=self.gradInput:mean(1)
      local gradInput_normPerDim=torch.norm(self.gradInput,1,1)/self.gradInput:size(1)
      
      local gradOutput_mean=gradOutput:mean(1)
      local gradOutput_normPerDim=torch.norm(gradOutput,1,1)/gradOutput:size(1)
     
     
      print('debug_batchModule--gradInput_mean:') 
      print(gradInput_mean)
      print('debug_batchModule--gradInput_normPerDim:') 
      print(gradInput_normPerDim)    
      print('debug_batchModule--gradOutput_mean:') 
      print(gradOutput_mean)    
      print('debug_batchModule--gradOutput_normPerDim:') 
      print(gradOutput_normPerDim)        
      
      
    --  print('-------------gradOuput----------')
    --  print(gradOutput)
    --  print('--------------gradInput-------')
     --  print(self.gradInput)
   end
    if self.debug_detailInfo then
    print('-----------------gradInput:-----------')
     print(self.gradInput[{{1,20},{}}]) 
   end
   self.count=self.count+1 --the ending of all the operation in this module
   
  -- print('-----------BN:gradInput-------------')
  -- print(self.gradInput)
   
   return self.gradInput
end

function BatchLinear_FIM:setTrainMode(isTrain)
  if isTrain ~= nil then
      assert(type(isTrain) == 'boolean', 'isTrain has to be true/false')
      self.train = isTrain
  else
    self.train=true  

  end
end


function BatchLinear_FIM:accGradParameters(input, gradOutput, scale)
   if self.affine then
      scale = scale or 1.0
      self.buffer2:resizeAs(self.normalized):copy(self.normalized)
      self.buffer2:cmul(gradOutput)
      self.buffer:sum(self.buffer2, 1) -- sum over mini-batch
      self.gradWeight:add(scale, self.buffer)
      self.buffer:sum(gradOutput, 1) -- sum over mini-batch
      self.gradBias:add(scale, self.buffer)
   end
end

