import tensorflow as tf
import numpy as np
import os
import sys
import time
from PIL import Image
from argparse import ArgumentParser
import model

ndomains = 8

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=str, dest="input_file", required=True, help="input file path")
parser.add_argument("-s", "--size", type=str, dest="size", required=False, help="input image size (WxH)")
parser.add_argument("-d", "--domain", type=str, dest="domain", required=False, help="target domains (20 comma separated float values in [-1, 1], ordered as follows: BlackHair,HvyMakeup,MouthSltOpen,NarrowEyes,NoBeard,Smiling,StraightHair,WearLipstick", default="1,-1,-1,-1,1,1,1,0")
args = parser.parse_args()

graph_def = tf.GraphDef.FromString(open("star_gan_%s.pb" % args.size, "rb").read())
graph = tf.Graph()
with graph.as_default():
  tf.import_graph_def(graph_def, name="")
sess = tf.InteractiveSession(graph=graph)
run_options = tf.RunOptions(output_partition_graphs=True)
run_metadata = tf.RunMetadata()

domain_labels = [
    "BlackHair", "HvyMakeup", "MouthSltOpen", "NarrowEyes", "NoBeard", "Smiling", "StraightHair", "WearLipstick"
]

def convert_png():
    filebase = args.input_file.split(os.sep)[-1].split(".")[0]
    original_image = Image.open(args.input_file).convert("RGB")
    original_image = np.asarray(original_image, dtype=np.float32)
    original_image = (original_image-128) / 128.0
    input_domain = np.array([float(d) for d in args.domain.split(",")])
    disc_domain = sess.run(["star_gan/discriminator/Sigmoid_1:0"], feed_dict = {"star_gan/input_image:0": [original_image]}, run_metadata=run_metadata, options=run_options)
    disc_domain = np.squeeze(disc_domain)
    target_domain = np.clip(disc_domain+input_domain, 0, 1)
    gen_image = sess.run(["star_gan/generator/Tanh:0"], feed_dict = {"star_gan/input_image:0": [original_image], "star_gan/input_domain:0": [target_domain]}, run_metadata=run_metadata, options=run_options)
    gen_image = tf.cast(tf.clip_by_value(gen_image[0]*128.0+128.0, 0, 255), tf.uint8)
    gen_image = np.reshape(gen_image.eval(session=sess), [gen_image.shape[1], gen_image.shape[2], gen_image.shape[3]])
    gen_image = Image.fromarray(gen_image, "RGB")
    gen_image.save(filebase + "_gen.png")

convert_png()
