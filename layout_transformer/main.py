import os
import argparse
import torch
from dataset import MNISTLayout, JSONLayout, ADE20KDataset
from evaluator import Evaluate, EvalConfig, Evaluate_2
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed
from sample_cofig import SampleConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="layout", help="experiment name")
    parser.add_argument("--log_dir", default="/home/hepe00001/Desktop/neuro_explicit/generative_diffusion/DeepLayout/layout_transformer/logs/", help="/path/to/logs/dir")

    # MNIST options
    parser.add_argument("--data_dir", default=None, help="/path/to/mnist/data")
    parser.add_argument("--threshold", type=int, default=16, help="threshold for grayscale values")

    # COCO/PubLayNet options
    parser.add_argument("--train_json", default="./instances_train.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="./instances_val.json", help="/path/to/val/json")

    # ade20K options
    parser.add_argument("--ade_background", default="json", help="an ADE20K background category")

    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--sample', action='store_true', help="sample only")    
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, args.exp)
    samples_dir = os.path.join(log_dir, "samples")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA is available. Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA is not available, running on CPU.")

    # check if we are doing sampling 
    if not args.sample:
        # MNIST Testing
        if args.data_dir == "MNIST":
            train_dataset = MNISTLayout(args.log_dir, train=True, threshold=args.threshold)
            valid_dataset = MNISTLayout(args.log_dir, train=False, threshold=args.threshold,
                                        max_length=train_dataset.max_length)
        # ADE20K   
        elif args.data_dir is not None:
            train_dataset = ADE20KDataset(args.data_dir)
            valid_dataset = ADE20KDataset(args.data_dir)

        # COCO and PubLayNet
        else:
            train_dataset = JSONLayout(args.train_json)
            valid_dataset = JSONLayout(args.val_json, max_length=train_dataset.max_length)

    else:
        sample_config = SampleConfig()
    
    # setup model
    mconf = GPTConfig(sample_config.vocab_size, sample_config.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
    model = GPT(mconf)

    if args.evaluate:
        evalconf = EvalConfig(max_epochs=args.epochs,
                                batch_size=args.batch_size,
                                ckpt_dir=ckpt_dir,
                                samples_dir=samples_dir,
                                sample_every=args.sample_every)
        
        # Inference mode: load the checkpoint and perform sampling
        if ckpt_dir is None:
            raise ValueError("Please provide a checkpoint path for evaluation.")
        
        print(f"Loading model from checkpoint: {ckpt_dir}")
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "checkpoint.pth"), map_location=device))
        
        evaluate = Evaluate(model, valid_dataset, evalconf, args)
        evaluate.eval()

    elif args.sample:
        evalconf = EvalConfig(max_epochs=args.epochs,
                                batch_size=args.batch_size,
                                ckpt_dir=ckpt_dir,
                                samples_dir=samples_dir,
                                sample_every=args.sample_every)
        
        # Inference mode: load the checkpoint and perform sampling
        if ckpt_dir is None:
            raise ValueError("Please provide a checkpoint path for evaluation.")
        
        print(f"Loading model from checkpoint: {ckpt_dir}")
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "checkpoint.pth"), map_location=device))

        sample_config = SampleConfig()
        evaluate = Evaluate_2(model, sample_config, evalconf, args)
        evaluate.eval()


    else:
        # Training mode: setup trainer and start training
        tconf = TrainerConfig(max_epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr_decay=args.lr_decay,
                                learning_rate=args.lr * args.batch_size,
                                warmup_iters=args.warmup_iters,
                                final_iters=args.final_iters,
                                ckpt_dir=ckpt_dir,
                                samples_dir=samples_dir,
                                sample_every=args.sample_every)
        trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
        trainer.train()
