cfg = {
    'nn_type'           : 'LeNet5',
    'normalize'         : True,
    'ds_name'           : 'MNIST',
    'batch_size'        : 4,
    'epochs'            : 50,
    'lr'                : 500*(60000**(-2)),
    'opacus'            : False,
    'scheduler_type'    : 'Cosine',
    'clip_val'          : -1,
    'save_model'        : True,
    'cuda_device_id'    : -1,
    'delta'             : 10**-5
}
