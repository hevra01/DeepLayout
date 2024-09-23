import json
import torch
import wandb
from prepare_for_visualization import process_all_images
from utils import sample

ade20K_eval_background = ["dining room", "bathroom", "airport airport", "airport terminal", "kitchen", 
                       "arcade", "bedroom", "living room", "classroom", "hospital room", 
                       "martial arts gym", "gym shoe", "gym machine"]


class EvalConfig:
    # optimization parameters
    max_epochs = 100
    batch_size = 64

    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers = 4  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Evaluate:
    def __init__(self, model, valid_dataset, evalConfig, args):
        self.model = model
        self.valid_dataset = valid_dataset
        self.evalConfig = evalConfig
        self.args = args
        
        print("Using wandb")
        wandb.login(key="aabe3a9de8b348d83b37bd4d1cbbdcd366f55c9e")
        wandb.init(project='LayoutTransformer', name=args.exp)
        wandb.config.update(args)
    
    def eval(self):
        self.model.eval()

        if self.args.ade_background is not None:
            with open('ade20K_labels.json', 'r') as file:
                ade_labels = json.load(file)
            
            # convert the background labels so that it can be used in the model
            labels = [ade_labels[backgr_label] for backgr_label in ade20K_eval_background]
            print(labels)
            labels = [self.valid_dataset.shifted_label(label) for label in labels]
            print(labels)
            # Get the bos_token
            bos_token = self.valid_dataset.bos_token
            print(bos_token)

            # get the bounding box for the background, which is fixed to cover the entire scene 
            bounding_box = self.valid_dataset.background.flatten()
            print(bounding_box)
            print(self.valid_dataset.max_length)

            # Create a tensor of shape [len(labels), 6]
            init_condition = torch.stack([
                torch.cat((torch.tensor([bos_token, label]), bounding_box))
                for label in labels
            ])
            print(init_condition)

        else:
            # Create a tensor of shape [4, 1] with each element having the value of 'bos'
            init_condition = torch.full((4, 1), self.valid_dataset.bos_token)
            #init_condition = [self.valid_dataset[i][0][:6] for i in range(4)]
            #init_condition = torch.stack(init_condition) 
        

        # samples - random
        layouts = sample(self.model, init_condition, steps=self.valid_dataset.max_length,
                                    temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        sample_random_layouts = [self.valid_dataset.render(layout) for layout in layouts]

        # Process all images and save as JSON
        process_all_images(layouts, self.valid_dataset, output_dir="./json_samples/random/1_tokens/")

        # samples - deterministic
        layouts = sample(self.model, init_condition, steps=self.valid_dataset.max_length,
                            temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
        sample_det_layouts = [self.valid_dataset.render(layout) for layout in layouts]
        
        # Process all images and save as JSON
        process_all_images(layouts, self.valid_dataset, output_dir="./json_samples/deterministic/1_tokens/")

        wandb.log({
                        "sample_random_layouts": [wandb.Image(pil, caption=f'bos_sample_random_{i:02d}.png')
                                                for i, pil in enumerate(sample_random_layouts)],
                        "sample_det_layouts": [wandb.Image(pil, caption=f'bos_sample_det_{i:02d}.png')
                                            for i, pil in enumerate(sample_det_layouts)],
                    }, step=self.valid_dataset.max_length)
        
