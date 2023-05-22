



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d

import _util.training_v1 as utraining


#################### hparams ####################

# specify hyperparameters
args = Dict()
args.base = Dict(
    name=fnstrip(__file__),
)
args.load = Dict(
)
args.prep = Dict(
    size=21,
    bs=64,
    augment_dilate_mask=(1,3),
)
args.model = Dict(
    # images
    patch_size=9,

    # generator
    gen_depth=6,
    gen_width=32,
    gen_batchnorm=True,
    gen_use_hull=False,
    gen_mask_input=True,

    # discriminator
    dis_depth=4,
    dis_width=16,
    dis_batchnorm=True,
    dis_use_hull=False,
    
    # learning
    lerp_output=True,
    label_smoothing=0.8,
    lr_gen=0.001,
    lr_dis=0.001,
    lambda_l1=1.0,
    lambda_adv=1.0,

    # boilerplate
    precision=16,
    gradient_clip_val=None,
    checkpoint=Dict(
        save_top_k=-1,
        every_n_epochs=10,
        # filename='{epoch:04d}',
    ),
)
args.train = Dict(
    gpus=torch.cuda.device_count(),
    max_bs_per_gpu=64,
    num_workers='max',
    max_epochs=1e6,

    loggers=[
        # 'wandb',
        'tensorboard',
    ],
    debug=Dict(
        check_val_every_n_epoch=args.model.checkpoint.every_n_epochs,
        # fast_dev_run=True,
        # log_every_n_steps=1,
        # limit_train_batches=4,
        # limit_val_batches=4,
        # limit_test_batches=4,
    ),
)


#################### train ####################

exec(f'import _train.img2img.datasets.{fnstrip(__file__).split("_")[0]} as module_dataset')
exec(f'import _train.img2img.models.{fnstrip(__file__).split("_")[1]} as module_model')

args_user = copy.deepcopy(args)
args = module_model.Model.update_args(args)

def _create_trainer():
    return pl.Trainer(
        default_root_dir=mkdir(f'./_train/img2img/runs/{args.base.name}'),

        gpus=args.train.gpus,
        precision=args.model.precision,
        accumulate_grad_batches=utraining.infer_batch_size(
            args.prep.bs, args.train.gpus, args.train.max_bs_per_gpu,
        )['accumulate_grad_batches'],
        max_epochs=args.train.max_epochs,
        **args.train.debug,

        callbacks=[pl.callbacks.ModelCheckpoint(
            **args.model.checkpoint,
            dirpath=mkdir(f'./_train/img2img/runs/{args.base.name}/checkpoints'),
            save_last=True,
        )],
        logger=[
            *([pl.loggers.TensorBoardLogger(
                mkdir(f'./_train/img2img/runs/{args.base.name}/logs'),
                name='tensorboard',
                version=0,
                log_graph=False,
                default_hp_metric=True,
                prefix='',
            )] if 'tensorboard' in args.train.loggers else []),
            # *([utraining.logger_wandb(args, im)] if 'wandb' in args.train.loggers else []),
        ] if not args.train.debug.fast_dev_run else False,
        
        detect_anomaly=True,
        strategy=pl.plugins.training_type.ddp.DDPPlugin(find_unused_parameters=True),
    )

if __name__=='__main__':
    model = module_model.Model(args=args)
    dm = module_dataset.Datamodule(args=args)
    trainer = _create_trainer()
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=f'./_train/img2img/runs/{args.base.name}/checkpoints/last.ckpt' \
            if os.path.isfile(f'./_train/img2img/runs/{args.base.name}/checkpoints/last.ckpt') \
            else None,
    )




