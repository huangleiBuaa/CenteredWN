--Centered  Weight Normalization
-- the code is based on the original implement of  weight normalizaiton on https://github.com/torch/nn/blob/master/WeightNorm.lua  and it's implementation: https://arxiv.org/pdf/1602.07868v3.pdf
local WeightNorm_Center, parent = torch.class("nn.WeightNorm_Center", "nn.Decorator")

function WeightNorm_Center:__init(module, outputDim, flag_adjustScale)
    -- this container will apply Weight Normalization to any module it wraps
    -- it accepts parameter ``outputDim`` that represents the dimension of the output of the weight
    -- if outputDim is not 1, the container will transpose the weight
    -- if the weight is not 2D, the container will view the weight into a 2D shape
    -- that is nOut x (nIn x kw x dw x ...)

    parent.__init(self, module)
    assert(module.weight)

    if module.bias then
        self.bias = module.bias
        self.gradBias = module.gradBias
    end
    if flag_adjustScale ~= nil then
        self.flag_adjustScale=flag_adjustScale
    else
        self.flag_adjustScale=false
    end
    self.gradWeight = module.gradWeight
    self.weight = module.weight

    self.outputDim = outputDim or 1

    -- track the non-output weight dimensions
    self.otherDims = 1
    for i = 1, self.weight:dim() do
        if i ~= self.outputDim then
            self.otherDims = self.otherDims * self.weight:size(i)
        end
    end

    -- view size for weight norm 2D calculations
    self.viewIn = torch.LongStorage({self.weight:size(self.outputDim), self.otherDims})

    -- view size back to original weight
    self.viewOut = self.weight:size()
    self.weightSize = self.weight:size()

    -- bubble outputDim size up to the front
    for i = self.outputDim - 1, 1, -1 do
        self.viewOut[i], self.viewOut[i + 1] = self.viewOut[i + 1], self.viewOut[i]
    end

    -- weight is reparametrized to decouple the length from the direction
    -- such that w = g * ( v / ||v|| )
    self.v = torch.Tensor(self.viewIn[1], self.viewIn[2])
    if self.flag_adjustScale then
        self.g = torch.Tensor(self.viewIn[1])
    else
        self.g = torch.Tensor(self.viewIn[1]):fill(1)
    end
    self._norm = torch.Tensor(self.viewIn[1])
    self._scale = torch.Tensor(self.viewIn[1])

    -- gradient of v
    self.gradV = torch.Tensor(self.viewIn)
        -- gradient of g
        self.gradG = torch.Tensor(self.viewIn[1]):zero()

    self:resetInit()
end

function WeightNorm_Center:permuteIn(inpt)
    local ans = inpt
    for i = self.outputDim - 1, 1, -1 do
        ans = ans:transpose(i, i+1)
       --print(ans)
    end
    return ans
end

function WeightNorm_Center:permuteOut(inpt)
    local ans = inpt
    for i = 1, self.outputDim - 1 do
        ans = ans:transpose(i, i+1)
    end
    return ans
end

function WeightNorm_Center:resetInit(inputSize, outputSize)
    self.v:normal(0, math.sqrt(2/self.viewIn[2]))
    if self.flag_adjustScale then
        -- self.g:norm(self.v, 2, 2)
         self.g:fill(1)
    end
    if self.bias then
        self.bias:zero()
    end
end

function WeightNorm_Center:evaluate()
    if not(self.train == false) then
        self:updateWeight()
        parent.evaluate(self)
    end
end

function WeightNorm_Center:updateWeight()
    -- view to 2D when weight norm container operates
    self.gradV:copy(self:permuteIn(self.weight))
    self.gradV = self.gradV:view(self.viewIn)

    -- ||w||
    self._norm:norm(self.v, 2, 2):pow(2):add(10e-5):sqrt()
    -- g * w / ||w||
    self.gradV:copy(self.v)
    self._scale:copy(self.g):cdiv(self._norm)
    self.gradV:cmul(self._scale:view(self.viewIn[1], 1)
                               :expand(self.viewIn[1], self.viewIn[2]))

    -- otherwise maintain size of original module weight
    self.gradV = self.gradV:view(self.viewOut)

    self.weight:copy(self:permuteOut(self.gradV))
end

function WeightNorm_Center:updateOutput(input)
    if not(self.train == false) then
     --  print('--------update weight---------------')
        self:updateWeight()
    end
    self.output:set(self.modules[1]:updateOutput(input))
    return self.output
end

function WeightNorm_Center:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    print(self.g)
    self.modules[1]:accGradParameters(input, gradOutput, scale)
    print(self.g)
    --print(self.gradWeight)
    self.weight:copy(self:permuteIn(self.weight))
    self.gradV:copy(self:permuteIn(self.gradWeight))
    self.weight = self.weight:view(self.viewIn)

    local norm = self._norm:view(self.viewIn[1], 1):expand(self.viewIn[1], self.viewIn[2])
    local scale = self._scale:view(self.viewIn[1], 1):expand(self.viewIn[1], self.viewIn[2])
    -- dL / dw * (w / ||w||)
        self.weight:copy(self.gradV)
        self.weight:cmul(self.v):cdiv(norm)
        self.gradG:sum(self.weight, 2)
    -- dL / dw * g / ||w||
    self.gradV:cmul(scale)

    -- dL / dg * (w * g / ||w||^2)
    self.weight:copy(self.v):cmul(scale):cdiv(norm)
    self.weight:cmul(self.gradG:view(self.viewIn[1], 1)
                               :expand(self.viewIn[1], self.viewIn[2]))

    -- dL / dv update
    self.gradV:add(-1, self.weight)

    self.gradV = self.gradV:view(self.viewOut)
    self.weight = self.weight:view(self.viewOut)
    self.gradWeight:copy(self:permuteOut(self.gradV))
end

function WeightNorm_Center:updateGradInput(input, gradOutput)
    self.gradInput:set(self.modules[1]:updateGradInput(input, gradOutput))
    return self.gradInput
end

function WeightNorm_Center:zeroGradParameters()
    self.modules[1]:zeroGradParameters()
    self.gradV:zero()
      self.gradG:zero()
end

function WeightNorm_Center:updateParameters(lr)
    self.modules[1]:updateParameters(lr)
    self.v:add(-lr, self.gradV)
    if self.flag_adjustScale then
       self.g:add(-lr, self.gradG)
    end
end

function WeightNorm_Center:parameters()
    if self.bias then
       if self.flag_adjustScale then
           return {self.v, self.g, self.bias}, {self.gradV, self.gradG, self.gradBias}
        else
           return {self.v,  self.bias}, {self.gradV,  self.gradBias}
        end
    else
       if self.flag_adjustScale then
            return {self.v, self.g}, {self.gradV, self.gradG}
        else
           return {self.v}, {self.gradV}
        end
    end
end

function WeightNorm_Center:write(file)
    -- Don't save weight and gradWeight since we can easily re-compute it from v
    -- and g.
    local weight = self.modules[1].weight
    local gradWeight = self.modules[1].gradWeight
    self.weight = nil
    self.gradWeight = nil
    self.modules[1].weight = nil
    self.modules[1].gradWeight = nil
    if not self.weightSize then
        self.weightSize = weight:size()
    end

    parent.write(self, file)

    self.modules[1].weight = weight
    self.modules[1].gradWeight = gradWeight
    self.weight = weight
    self.gradWeight = gradWeight
end

function WeightNorm_Center:read(file)
    parent.read(self, file)

    -- Re-compute weight and gradWeight
    if not self.weight then
        self.modules[1].weight = self.v.new(self.weightSize)
        self.modules[1].gradWeight = self.v.new(self.weightSize)
        self.weight = self.modules[1].weight
        self.gradWeight = self.modules[1].gradWeight
        self:updateWeight()
        self.gradWeight:copy(self:permuteOut(self.gradV))
    end
end
