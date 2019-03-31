import tensorflow as tf
import os
import json
import model
from argparse import ArgumentParser
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.saved_model import signature_constants

ndomains = 8
parser = ArgumentParser()
parser.add_argument("-s", "--size", type=str, dest="size", required=False, help="input image size (WxH) for protocol buffer")
args = parser.parse_args()
width, height = [int(x) for x in args.size.split("x")]

# def fp32_get_var(getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, *args, **kwargs):
    # storage_dtype = tf.float32 if trainable else dtype
    # variable = getter(name, shape, dtype=storage_dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, *args, **kwargs)
    # if trainable and dtype != tf.float32:
        # variable = tf.cast(variable, dtype)
    # return variable

with tf.Graph().as_default() as graph:
    star_gan = model.StarGAN(batch_size=1, image_shape=[height,width,3])
    with tf.Session() as sess:
        # with tf.variable_scope("star_gan", custom_getter=fp32_get_var):
        with tf.variable_scope("star_gan"):
            #tf.global_variables_initializer().run()
            input_image_tf = tf.placeholder(tf.float32, [1, height, width, 3], name="input_image")
            input_domain_tf = tf.placeholder(tf.float32, [1, ndomains], name="input_domain")
            gen_image_tf = star_gan.generate(input_image_tf, input_domain_tf)
            output_image_tf = tf.identity(gen_image_tf, 'output_image')
            _, disc_domain_tf = star_gan.discriminate(tf.image.resize_bicubic(input_image_tf, [320, 320]))
            target_domain = tf.clip_by_value(disc_domain_tf+input_domain_tf, 0, 1)
            # print(sess.graph.get_operations())
            # generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="(?!.*normalization.*)")
            generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="star_gan/generator/*|star_gan/discriminator/*")
            generator_vars_dict = dict([(var.op.name, var) for var in generator_vars])
            generator_saver = tf.train.Saver(generator_vars_dict)
            generator_saver.restore(sess, "./star_gan.ckpt")
            print()
            """
            graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ["star_gan/generator/Tanh", "star_gan/discriminator/Sigmoid_1"])
            tf.train.write_graph(graph_def, '.', 'star_gan_%s.pb' % args.size, as_text=False)
            """

            builder = tf.saved_model.builder.SavedModelBuilder('1')
            ipts = {'input_image': input_image_tf, "input_domain":input_domain_tf}
            ipts = {key: utils.build_tensor_info(tensor)
                      for key, tensor in ipts.items()}
            otpts = {'output_image' : output_image_tf}
            otpts = {key: utils.build_tensor_info(tensor)
                       for key, tensor in otpts.items()}
            
            signature_def = (tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=ipts,
                    outputs=otpts, method_name=signature_constants.PREDICT_METHOD_NAME))

            signature_def_key = (tf.saved_model.signature_constants.
                         DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            signature_def_map = {signature_def_key: signature_def}
            builder.add_meta_graph_and_variables(sess=sess,
                                                 tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map=signature_def_map)
            builder.save(as_text=False)
