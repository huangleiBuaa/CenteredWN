require 'nn'
require 'cunn'
require 'cudnn'
require '../module/spatial/cudnn_Spatial_Weight_CenteredBN'
require '../module/spatial/Spatial_Scaling'
local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end
  
local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(cudnn.Spatial_Weight_CenteredBN(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):noBias())
   --vgg:add(nn.Spatial_Scaling(nOutputPlane,_,_,nInputPlane))
   vgg:add(nn.Spatial_Scaling(nOutputPlane,opt.BNScale,true))
  vgg:add(backend.ReLU(true))

  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = backend.SpatialMaxPooling
local n=opt.base_hidden
ConvBNReLU(3,n)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(n,n*2)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(n*2,n*4)
ConvBNReLU(n*4,n*4)
vgg:add(MaxPooling(2,2,2,2))

ConvBNReLU(n*4,n*8)
ConvBNReLU(n*8,n*8)
vgg:add(MaxPooling(2,2,2,2))

-- In the last block of convolutions the inputs are smaller than
-- the kernels and cudnn doesn't handle that, have to use cunn
backend = nn
ConvBNReLU(n*8,n*8)
ConvBNReLU(n*8,n*8)
vgg:add(MaxPooling(2,2,2,2))
vgg:add(nn.View(n*8))

classifier = nn.Sequential()
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(n*8,n*8))
classifier:add(nn.BatchNormalization(n*8))
classifier:add(nn.ReLU(true))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(n*8,10))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
