"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torchvision

# import your model class
# import ...
from model import BoundingBoxesNN, RoadMapNN

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'IBM'
    team_number = 56
    round_number = 3
    team_member = ['Alexander Bienstock', 'Go Inoue', 'Shujaat Mirza']
    contact_email = 'afb383@nyu.edu, gi372@nyu.edu, msm622@nyu.edu'

    def __init__(self, model_file={'box': './model/4angle_boundingBoxNN_model_at_epoch_10.pt',
                                   'map': './model/roadMapNN_model_at_epoch_10.pt'}):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...
        # 
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.box_model = BoundingBoxesNN(device=self.device)
        self.box_model.load_model(model_file['box'])

        self.map_model = RoadMapNN(device=self.device)
        self.map_model.load_model(model_file['map'])

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        for batch_idx, sample in enumerate(samples):
            # [6, 3, 256, 306]
            samples = torchvision.utils.make_grid(sample, nrow=3, padding=0).unsqueeze(0).to(self.device)
            output = self.box_model.model(samples)
            prediction = self.box_model.predict(output) # batch X N X 2 X $
        return tuple([prediction[0]['boxes']])

        # return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
        for batch_idx, sample in enumerate(samples):
            # [6, 3, 256, 306]
            samples = torchvision.utils.make_grid(sample, nrow=3, padding=0).unsqueeze(0).to(self.device)
            # [3, 512, 918]
            # print(samples.shape, file=sys.stderr)
            output = self.map_model.model(samples)
            prediction = self.map_model.predict(output)
            # print(len(prediction), file=sys.stderr)
        return prediction
        # return torch.rand(1, 800, 800) > 0.5
