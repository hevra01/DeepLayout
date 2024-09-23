import json
import torch


class SampleConfig:
    def __init__(self,bos_token=None,eos_token=None,pad_token=None, standard_frame_width=None, 
                    standard_frame_height=None, max_length=None, size=None):
        
        self.vocab_size = 4516
        
        # these are the vocab indices after considering the bounding boxes and the class labels
        self.bos_token = 4513 
        self.eos_token = 4514
        self.pad_token = 4515

        self.standard_frame_width = 512
        self.standard_frame_height = 512

        self.max_length = 157 # this is the maximum number of tokens found in the dataset
        self.size = 512 # this is the bounding box limit 

        self.background = torch.tensor([[0, 0, self.standard_frame_width - 1, self.standard_frame_height - 1]])

        with open('ade20K_labels.json', 'r') as file:
            self.ade_labels = json.load(file)


        self.shift_by_self_size = {
                v: i + self.size for i, v in enumerate(self.ade_labels.values())
        }

        self.undo_shift = {
            v: k for k, v in self.shift_by_self_size.items()
        }
