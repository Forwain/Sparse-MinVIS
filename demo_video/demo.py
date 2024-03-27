# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import numpy as np
import tqdm

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config
from predictor import VisualizationDemo

import shutil

import imageio.v2 as imageio
import random

def setup_cfg(args):
	# load config from file and command-line arguments
	cfg = get_cfg()
	add_deeplab_config(cfg)
	add_maskformer2_config(cfg)
	add_maskformer2_video_config(cfg)
	add_minvis_config(cfg)
	cfg.merge_from_file(args.config_file)
	cfg.merge_from_list(args.opts)
	cfg.CONFIDENCE_THRESHOLD = args.threshold
	cfg.freeze()
	return cfg

def get_parser():
	parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
	parser.add_argument(
		"--config-file",
		default="configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml",
		metavar="FILE",
		help="path to config file",
	)
	parser.add_argument(
		"--input",
		help="directory of input video frames",
		required=True,
	)
	parser.add_argument(
		"--output",
		help="directory to save output frames",
		required=True,
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.1,
		help="Minimum score for instance predictions to be shown",
	)
	parser.add_argument(
		"--opts",
		help="Modify config options using the command-line 'KEY VALUE' pairs",
		default=[],
		nargs=argparse.REMAINDER,
	)
	return parser

def demo_on_dataset(demo, dataset_name, output_root):
	if dataset_name == "test_all":
		dataset_root = "../datasets/ytvis_2019/test/JPEGImages"
	elif dataset_name == "valid_all":
		dataset_root = "../datasets/ytvis_2019/valid/JPEGImages"
	if os.path.exists(output_root):
		shutil.rmtree(output_root)
	os.makedirs(output_root, exist_ok=True)
	# for video_name in random.sample(os.listdir(dataset_root), k=50):
	for video_name in os.listdir(dataset_root):
		frames_path = os.path.join(dataset_root, video_name)
		frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, '*.jpg')))
		frames_path.sort()
		os.makedirs(os.path.join(output_root, video_name), exist_ok=True)
		vid_frames = []
		for path in frames_path:
			img = read_image(path, format="BGR")
			vid_frames.append(img)

		start_time = time.time()
		with autocast():
			predictions, visualized_output = demo.run_on_video(vid_frames)
		logger.info(
			f"inference on video {video_name} with {len(frames_path)} frames in {time.time() - start_time:.2f}s"
			)

		# save frames all in one
		for path, _vis_output in zip(frames_path, visualized_output):
			out_filename = os.path.join(output_root, video_name, os.path.basename(path))
			_vis_output.save(out_filename)
		
		with autocast():
			visualized_output = demo.run_per_ins(vid_frames)

		# save frames per instance
		for i, ins_vis_output in enumerate(visualized_output):
			ins_out_filename = os.path.join(output_root, video_name, f"ins_{i}")
			os.makedirs(ins_out_filename, exist_ok=True)
			#/output/0dfc2c/1/
			jpg_files = []
			for (filename, frame_vis_output) in zip(frames_path, ins_vis_output):
				frame_out_filename = os.path.join(ins_out_filename, os.path.basename(filename))
				frame_vis_output.save(frame_out_filename)
				logger.info(f"save {frame_out_filename}")
				jpg_files.append(frame_out_filename)
			# save .gif
			images = [imageio.imread(filename) for filename in jpg_files]
			imageio.mimsave(os.path.join(ins_out_filename, "output.gif"), images)


def demo_on_video(demo, video_root, output_root):
	os.makedirs(output_root, exist_ok=True)
	
	frames_path = video_root
	frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, '*.jpg')))
	frames_path.sort()

	vid_frames = []
	for path in frames_path:
		img = read_image(path, format="BGR")
		vid_frames.append(img)

	start_time = time.time()
	with autocast():
		predictions, visualized_output = demo.run_on_video(vid_frames)
	logger.info(
		"detected {} instances per frame in {:.2f}s".format(
			len(predictions["pred_scores"]), time.time() - start_time
		)
	)

	# save frames all in one
	for path, _vis_output in zip(frames_path, visualized_output):
		out_filename = os.path.join(output_root, os.path.basename(path))
		_vis_output.save(out_filename)
	
	with autocast():
		visualized_output = demo.run_per_ins(vid_frames)

	# save frames per instance
	for i, ins_vis_output in enumerate(visualized_output):
		ins_out_filename = os.path.join(output_root, os.path.basename(os.path.dirname(path)), f"ins_{i}")
		os.makedirs(ins_out_filename, exist_ok=True)
		#/output/0dfc2c/1/
		jpg_files = []
		for (filename, frame_vis_output) in zip(frames_path, ins_vis_output):
			frame_out_filename = os.path.join(ins_out_filename, os.path.basename(filename))
			frame_vis_output.save(frame_out_filename)
			jpg_files.append(frame_out_filename)
		# save .gif
		images = [imageio.imread(filename) for filename in jpg_files]
		imageio.mimsave(os.path.join(ins_out_filename, "output.gif"), images)

if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	args = get_parser().parse_args()
	setup_logger(name="fvcore")
	logger = setup_logger()
	logger.info("Arguments: " + str(args))

	cfg = setup_cfg(args)
	demo = VisualizationDemo(cfg)

	if args.input == "test_all" or args.input == "valid_all":
		# inference on dataset
		demo_on_dataset(demo, args.input, args.output)
	else:
		# inference on single video
		demo_on_video(demo, video_root=args.input, output_root=args.output)
