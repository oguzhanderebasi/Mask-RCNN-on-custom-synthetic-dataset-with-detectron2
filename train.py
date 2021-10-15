# check pytorch installation:

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
assert torch.__version__.startswith("1.9")
"""## Get data"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import argparse
import json
import string

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

"""## Register data-set

In order to use a dataset with Detectron2 we need to register it. For more information check out the official documentation.
"""

parser = argparse.ArgumentParser(description="Segmentation with Mask-RCNN")

# Add arguments

parser.add_argument("-p_name", "--project_name", help="Project name")
parser.add_argument("-ddir", "--dataset_dir", help="Dataset direction that contains train.json and val.json in coco format.")
parser.add_argument("--num_worker", default=2, type=int, help="The num_workers attribute tells the data loader instance how many sub-processes to use for data loading.")
parser.add_argument("--ims_per_bach", default=2, type=int, help="Number of images per batch across all machines.")
parser.add_argument("--l_rate", default=0.00025, type=int, help="LearningRate")
parser.add_argument("-iter","--iteration", default=1000, type=int, help="Number of iteration")
parser.add_argument("--n_classes", type=int, help="Number of classes in the dataset")
parser.add_argument("--t_treshold", default=5, type=int, help="Set the testing treshold for this model")
parser.add_argument("-o_dir", "--output_dir", help="Output direction of trained model")


# Indicate end of argument definitions and parse args
args = parser.parse_args()

for d in ["train", "val"]:
    register_coco_instances(args.project_name + f"_{d}", {}, args.dataset_dir + f"/{d}.json", args.dataset_dir + f"/{d}")

print(args.project_name + f"_train")

dataset_dicts = DatasetCatalog.get(args.project_name + "_train")
part_metadata = MetadataCatalog.get(args.project_name + "_train")

for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=part_metadata, scale=0.5)
    v = v.draw_dataset_dict(d)
    plt.figure(figsize = (14, 10))

# if you work in colab env., you can use this 2 following lines
#    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#    plt.show()

'''
    #Save Metadata
'''
with open(args.output_dir + '/metadata.json', 'w') as outfile:
  json.dump(part_metadata.as_dict(), outfile)


"""## Train model

Now, let's fine-tune a pretrained MaskRCNN instance segmentation model on a dataset.
"""

cfg = get_cfg()
#cfg.MODEL.DEVICE = "cpu" # activate this if you can't use GPU.
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (args.project_name + "_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = args.num_worker
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = args.ims_per_bach
cfg.SOLVER.BASE_LR = args.l_rate
cfg.SOLVER.MAX_ITER = args.iteration
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.n_classes

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Commented out IPython magic to ensure Python compatibility.
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output

"""## Use model for inference

Now, we can perform inference on our validation set by creating a predictor object.
"""

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = (args.project_name + "_val", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = DatasetCatalog.get(args.project_name + "_val")
for d in random.sample(dataset_dicts, 6):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=part_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))

# if you work in colab env., you can use this 2 following lines
#    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#    plt.show()

'''  
We can also evaluate its performance using AP metric implemented in COCO API
'''
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator(args.project_name + "_val", output_dir=args.output_dir)   #Give the output direction
val_loader = build_detection_test_loader(cfg, args.project_name + "_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))