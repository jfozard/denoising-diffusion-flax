import ml_collections

def get_config():

  config = ml_collections.ConfigDict()


  # wandb
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.entity = None
  wandb.project = "leaf-unet"
  wandb.job_type = "training"
  wandb.name = None 
  wandb.log_train = True
  wandb.log_sample = True
  wandb.log_model = True
  

  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_train_steps = 700000
  training.log_every_steps = 100
  training.loss_type = 'l2'
  training.half_precision = False
  training.save_and_sample_every = 1000
  training.num_sample = 4


  # ema
  config.ema = ema = ml_collections.ConfigDict()
  ema.beta = 0.995
  ema.update_every = 10
  ema.update_after_step = 100
  ema.inv_gamma = 1.0
  ema.power = 2 / 3
  ema.min_value = 0.0

  #
  config.unet = unet = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset ='jf_leaves'
  data.batch_size = 2
  data.cache = False
  data.image_size = 64
  data.channels = 1


  # model
  config.model = model = ml_collections.ConfigDict()
  model.dim = 64
  model.dim_mults = (1, 2, 4, 8)
  
  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 8e-5
  optim.beta1 = 0.9
  optim.beta2 = 0.99
  optim.eps = 1e-8

  config.seed = 42

  return config


