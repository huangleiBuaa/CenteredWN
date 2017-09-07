local Linear_Weight_CenteredBN_Row, parent = torch.class('nn.Linear_Weight_CenteredBN_Row', 'nn.Module')

function Linear_Weight_CenteredBN_Row:__init(inputSize,outputSize, flag_adjustScale,init_flag)
   parent.__init(self)

   self.weight = torch.Tensor( outputSize,inputSize) 
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   
  if flag_adjustScale ~= nil then
      self.flag_adjustScale= flag_adjustScale
   else
      self.flag_adjustScale= false
   end 
  if init_flag ~= nil then
      self.init_flag = init_flag
   else
      self.init_flag = 'RandInit'
   end 

    self.g=torch.Tensor(outputSize):fill(1)
    if self.flag_adjustScale then
     self.gradG=torch.Tensor(outputSize)
     self.gradBias = torch.Tensor(outputSize)
     self.bias = torch.Tensor(outputSize):fill(0)
    end 
    self:reset()

end



function Linear_Weight_CenteredBN_Row:reset(stdv)
    if self.init_flag=='RandInit' then
        self:reset_RandInit(stdv)
    elseif self.init_flag=='OrthInit' then
        self:reset_orthogonal(stdv)
    end
    return self
end

function Linear_Weight_CenteredBN_Row:reset_RandInit(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
        -- self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
     -- self.bias:uniform(-stdv, stdv)
   end
end

function Linear_Weight_CenteredBN_Row:reset_orthogonal()
    local initScale = 1.1 -- math.sqrt(2)

    local M1 = torch.randn(self.weight:size(1), self.weight:size(1))
    local M2 = torch.randn(self.weight:size(2), self.weight:size(2))

    local n_min = math.min(self.weight:size(1), self.weight:size(2))

    -- QR decomposition of random matrices ~ N(0, 1)
    local Q1, R1 = torch.qr(M1)
    local Q2, R2 = torch.qr(M2)

    self.weight:copy(Q1:narrow(2,1,n_min) * Q2:narrow(1,1,n_min)):mul(initScale)

   -- self.bias:zero()
end


function Linear_Weight_CenteredBN_Row:updateOutput(input)

  if input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.output:resize(nframe, n_output)
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()      
      self.addBuffer:resize(nframe):fill(1)

      self.mean=self.mean or input.new()
      self.std=self.std or input.new()

      
      self.W=self.W or input.new()
      self.W_hat=self.W_hat or input.new()
      self.W:resizeAs(self.weight)

      self.mean:mean(self.weight, 2) 
      self.weight:add(-self.mean:expand(n_output,n_input))
      
       self.std:resize(n_output,1):copy(self.weight:norm(2,2)):pow(-1)
      
      
      
       self.W_hat:resizeAs(self.weight):copy(self.weight):cmul(self.std:expand(n_output,n_input))
      self.W:copy(self.W_hat):cmul(self.g:view(n_output,1):expand(n_output,n_input))
      self.output:addmm(0, self.output, 1, input, self.W:t())
      if self.flag_adjustScale then
         self.output:addr(1, self.addBuffer, self.bias)
       end 
  else
      error('input must be vector or matrix')
   end
   
   return self.output
end

function Linear_Weight_CenteredBN_Row:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      
      
     if input:dim() == 2 then
         
         self.gradInput:addmm(0, 1, gradOutput, self.W)    
         
     else
      error('input must be vector or matrix')
     end
  
      
      return self.gradInput
   end
end

function Linear_Weight_CenteredBN_Row:accGradParameters(input, gradOutput, scale)
   --   if self.flag_inner_lr then
   --     scale = self.scale or 1.0
   --   else
        scale =scale or 1.0
  --    end
   if input:dim() == 2 then
      local n_output=self.weight:size(1)
      local n_input=self.weight:size(2)
      self.gradW=self.gradW or input.new()
      self._scale=self._scale or input.new()
      self._scale:resizeAs(self.std):copy(self.std):cmul(self.g)
      self.gradW:resize(gradOutput:size(2),input:size(2))
      self.gradW:mm(gradOutput:t(), input)  --dL/dW

      
      self.gradWeight:cmul(self.W_hat, self.gradW)
       self.mean:sum(self.gradWeight,2)   
      self.gradWeight:copy(-self.W_hat):cmul(self.mean:expand(n_output,n_input))
   
      self.mean:mean(self.gradW,2) 
      self.gradWeight:add(self.gradW):add(-self.mean:expand(n_output,n_input))
        
        self.gradWeight:cmul(self._scale:expand(n_output,n_input))
    --print(self.g)
    --print(self.bias)

     if self.flag_adjustScale then 
        self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
        self.W_hat:cmul(self.gradW)
        self.gradG:sum(self.W_hat,2)
    end
   else
      error('input must be vector or matrix')
   end
   

end

function Linear_Weight_CenteredBN_Row:parameters()

    if self.flag_adjustScale then 
        return {self.weight, self.g, self.bias}, {self.gradWeight, self.gradG, self.gradBias}
     else
        return {self.weight}, {self.gradWeight}

    end 
end

-- we do not need to accumulate parameters when sharing
Linear_Weight_CenteredBN_Row.sharedAccUpdateGradParameters = Linear_Weight_CenteredBN_Row.accUpdateGradParameters


function Linear_Weight_CenteredBN_Row:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
