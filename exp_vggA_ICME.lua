require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('compare the Decorelated BatchNormalizaiton method with baselines on MLP architechture')
cmd:text()
cmd:text('Options')

cmd:option('-model','vggA_plain','the methods')
cmd:option('-max_epoch',80,'maximum number of iterations')
cmd:option('-epoch_step',20000,'epoch step: no lr annealing if it is larger than the maximum')
cmd:option('-save',"../0_experiment_result/log_Conv_6Final" ,'subdirectory to save logs')
cmd:option('-batchSize',256,'the number of examples per batch')

cmd:option('-optimization','simple','the methods: options:simple,rms,adagrad,adam')
cmd:option('-learningRate',1,'initial learning rate')
---for simple (sgd)--------
cmd:option('-lrD_k',2000,'exponential learning rate decay, and each lrD_k iteration the learning rate become half')
cmd:option('-weightDecay',0.0005,'weight Decay for regularization')
cmd:option('-momentum',0.9,'momentum')
------------for rms/ agagrad-------
cmd:option('-rms_alpha',0,'the rate of the rms method')

cmd:option('-m_perGroup',64,'the number of per group')
cmd:option('-topK',12,'for DBN_PK method, scale the topK eigenValue')
cmd:option('-eig_epsilo',1e-3,'for DBN_PEP method, scale the eigenValue larger eig_epsilo')
cmd:option('-base_hidden',64,'for DBN_PEP method, scale the eigenValue larger eig_epsilo')

cmd:option('-weight_debug',0,'0 indicates not debug weight; 1 indicates debug Global weight and GradW; 2 indicates add observe per module')
cmd:option('-step_WD',1,'the step to debug the weights')
cmd:option('-BNScale',1,'the initial value for BN scale')
cmd:option('-T',195,'the interval to update the weight for SVB method')
cmd:option('-orth_flag',false,'is orthogonal initialization only for SVB method')


cmd:option('-seed',1,'the step to debug the weights')

cmd:text()

-- parse input params
 opt = cmd:parse(arg)

 opt.rundir = cmd:string('console/result_', opt, {dir=true})
 paths.mkdir(opt.rundir)

-- -- create log file
 cmd:log(opt.rundir .. '/log', opt)

cutorch.manualSeed(opt.seed)
print(c.blue'==>' ..' configuring optimizer')

if opt.optimization == 'simple' then
  opt.optimState = {
    learningRate =opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = learningRateDecay
  }
 optimMethod = optim.sgd
elseif opt.optimization == 'adagrad' then
  opt.optimState = {
    learningRate = opt.learningRate,
  }
 optimMethod = optim.adagrad
elseif opt.optimization == 'rms' then
  opt.optimState = {
    learningRate = opt.learningRate,
    alpha=opt.rms_alpha 
  }
 optimMethod = optim.rmsprop
elseif opt.optimization == 'adam' then
  opt.optimState = {
    learningRate = opt.learningRate
  }
 optimMethod = optim.adam
else
  error('Unknown optimizer')
end

print(opt)
torch.manualSeed(opt.seed)    -- fix random seed so program runs the same every time
threadNumber=4
torch.setnumthreads(threadNumber)
torch.setdefaulttensortype('torch.FloatTensor')
do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output = input
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(2).updateGradInput = function(input) return end
print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load '/home/huanglei/torch_work/dataset/cifar_provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)

log_name=opt.model..'_'..opt.optimization..'_lr'..opt.learningRate..'_g'..opt.m_perGroup..'_mm'..opt.momentum..'_'..opt.weightDecay..'_'..opt.lrD_k..'.log'

testLogger = optim.Logger(paths.concat(opt.save, log_name))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()




function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then opt.optimState.learningRate = opt.optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

 -- local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)
 

       print(string.format("Iter: %6s,  loss = %6.6f", iteration,f))

   --   print(string.format("minibatches processed: %6s, loss = %6.6f", iteration, f))
      losses[#losses + 1] = f
      timeCosts[#timeCosts+1]=torch.toc(start_time)
      print(string.format("time Costs = %6.6f", timeCosts[#timeCosts]))
  

      iteration=iteration+1
      return f,gradParameters
    end
   -----learning rate schedule---------------
   local k=torch.log(2)/opt.lrD_k
   local lr_scale=torch.exp(-k*iteration)	
   opt.optimState.learningRate=opt.learningRate* lr_scale
   print('learning Rate:'..opt.optimState.learningRate)
   optimMethod (feval, parameters, opt.optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(start_time)))

  train_acc = confusion.totalValid * 100
  train_accus[#train_accus+1]=train_acc
  confusion:zero()
  epoch = epoch + 1
end


 


function test()

 model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
   test_accus[#test_accus+1]=confusion.totalValid * 100
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()
   end

  confusion:zero()
end

iteration=0
losses={}
timeCosts={}
train_accus={}
test_accus={}
start_time=torch.tic()
train_times={}
test_times={}


for i=1,opt.max_epoch do
  local function t(f) local s = torch.Timer();f() return  s:time().real end

  local  train_time = t(train)
 train_times[#train_times+1]=train_time
 print('train Time:'..train_time) 


  local  test_time = t(test) 
  test_times[#test_times+1]=test_time
  print('test Time:'..test_time)

end

results={}
results.opt=opt
results.losses=losses
results.train_accus=train_accus
results.test_accus=test_accus
results.train_times=train_times
results.test_times=test_times


--results.timeCosts=timeCosts
--results.testLogger=testLogger
results.confusion=confusion
results.opt.optimState.dfdx=nil
torch.save('result_ICML_'..opt.model..'_'..opt.optimization..'_lr'..opt.learningRate..'_g'..opt.m_perGroup..'_mm'..opt.momentum
..'_'..opt.weightDecay
..'_BH'..opt.base_hidden..'_BNS'..opt.BNScale
..'_'..opt.lrD_k..'_seed'..opt.seed
..'.dat',results) 
