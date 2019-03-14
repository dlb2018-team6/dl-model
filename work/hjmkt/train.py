import numpy as np
import tensorflow as tf
from numpy import inf
import glob
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance
from argparse import ArgumentParser
import model

BATCH_SIZE = 6
WIDTH = 320
HEIGHT = 320

domain_mask = [8,18,21,23,24,31,32,36]
ndomains = 8
nepochs = 100
parser = ArgumentParser()
parser.add_argument("--image-data", type=str, dest="image_data", required=True, help="image dataset path")
parser.add_argument("--label-data", type=str, dest="label_data", required=True, help="label dataset path")
parser.add_argument("-r", "--learning-rate", type=float, dest="learning_rate", required=False, help="learning rate", default=1e-4)
args = parser.parse_args()

def preproc_dataset(path, image_width, image_height):
    images = []
    for i in range(len(path)):
        image = Image.open(path[i])
        image = image.convert("RGB")
        width, height = image.size[0], image.size[1]
        if height>width:
            image = np.asarray(image)
            if height>width*3//2:
                pad = (height-width*3//2+1) // 2
                if pad>0:
                    left_border = np.tile(image[:, :1, :], (1, pad, 1))
                    right_border = np.tile(image[:, -1:, :], (1, pad, 1))
                    image = np.concatenate([left_border, image, right_border], axis=1)
            image = Image.fromarray(image, "RGB")
            scale = np.random.uniform(320, 480+1) / width
            width = np.ceil(width*scale).astype(int)
            height = np.ceil(height*scale).astype(int)
            image = image.resize((width, height), Image.LANCZOS)
        else:
            image = np.asarray(image)
            if width>height*3//2:
                pad = (width-height*3//2+1) // 2
                if pad>0:
                    top_border = np.tile(image[:1, :, :], (pad, 1, 1))
                    bottom_border = np.tile(image[-1:, :, :], (pad, 1, 1))
                    image = np.concatenate([top_border, image, bottom_border], axis=0)
            image = Image.fromarray(image, "RGB")
            scale = np.random.uniform(320, 480+1) / height
            width = np.ceil(width*scale).astype(int)
            height = np.ceil(height*scale).astype(int)
            image = image.resize((width, height), Image.LANCZOS)

        if width>image_width and height>image_height:
            h_offset = random.randrange(height-image_height)
            w_offset = random.randrange(width-image_width)
            image= image.crop((w_offset, h_offset, w_offset+image_width, h_offset+image_height))
        else:
            image = image.resize((image_width, image_height), Image.LANCZOS)
        image = image.convert("RGB")
        image = np.asarray(image, np.float32)
        image = (image-128) / 128.0
        images.append(image)
    return images

def get_dataset_path(base_image_path, base_label_path):
    images = glob.glob(base_image_path + "/img_celeba/*")
    image_basenames = list(map(lambda path: path.split(os.sep)[-1].split(".")[0], images))
    images = list(map(lambda filename: base_image_path + "/img_celeba/" + filename + ".jpg", image_basenames))
    domain_path = base_label_path + "/list_attr_celeba.csv"
    domains = []
    image_names = []
    with open(domain_path) as f:
        line = f.readline()
        line = f.readline()
        while line:
            data = line.split(",")
            image_name = data[0]
            domain = [1.0 if d.replace('\n', '')=="1" else 0.0 for d in data[1:]]
            domain = np.array(domain)[np.array(domain_mask)]
            domains.append(domain)
            image_names.append(image_name)
            line = f.readline()
    images = sorted(images)
    return np.array(images), np.array(domains),np.array(image_names)

def get_dataset(hq):
    hq_images = preproc_dataset(hq, WIDTH, HEIGHT)
    hq_images = np.asarray(hq_images, dtype=np.float32)
    return hq_images

sess = tf.InteractiveSession()
lq_sess = tf.Session()
dataset = get_dataset_path(args.image_data, args.label_data)
star_gan = model.StarGAN(batch_size=BATCH_SIZE, image_shape=[HEIGHT,WIDTH,3])

def tf_log(x):
    x = tf.maximum(x, 1e-10)
    return tf.log(x)

def setup():
    Z_image = tf.placeholder(tf.float32, [star_gan.batch_size, star_gan.image_shape[0], star_gan.image_shape[1], star_gan.image_shape[2]], name="input_image")
    Z_original_domain = tf.placeholder(tf.float32, [star_gan.batch_size, star_gan.domains], name="input_original_domain")
    Z_target_domain = tf.placeholder(tf.float32, [star_gan.batch_size, star_gan.domains], name="input_target_domain")
    original = tf.image.random_flip_left_right(Z_image)
    gen = star_gan.generate(original, Z_target_domain)
    reconst = star_gan.generate(gen, Z_original_domain)

    p_original, dom_original = star_gan.discriminate(original)
    identity = star_gan.generate(original, dom_original)
    p_original = tf.clip_by_value(p_original, 1e-10, 1)
    dom_original = tf.clip_by_value(dom_original, 1e-10, 1)
    p_gen, dom_gen = star_gan.discriminate(gen)
    p_gen = tf.clip_by_value(p_gen, 1e-10, 1)
    dom_gen = tf.clip_by_value(dom_gen, 1e-10, 1)
    def gradient_penalty(r, f):
        epsilon = tf.random_uniform(r.shape, minval=-1, maxval=1, dtype=tf.float32)
        m, v = tf.nn.moments(r, axes=[0, 1, 2, 3])
        sd = tf.sqrt(v)
        delta = sd/2 * epsilon
        alpha = tf.random_uniform([BATCH_SIZE, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
        hat = tf.clip_by_value(r+alpha*delta, -1, 1)
        logit, _ = star_gan.discriminate(hat)
        grad = tf.gradients(logit, hat)[0]
        grad = tf.norm(tf.layers.flatten(grad), axis=1)
        return tf.reduce_mean(grad) * 10
        # return tf.contrib.gan.losses.wargs.wasserstein_gradient_penalty(
                # r, f, None, lambda x,y: star_gan.discriminate(x), "discriminator"
        # ) * 10

    gp_loss = gradient_penalty(Z_image, gen)

    image_disc_loss = tf.reduce_mean(
        -tf.reduce_mean(tf_log(p_original) + tf_log(tf.ones(star_gan.batch_size, tf.float32) - p_gen), axis=1)
    ) + gp_loss

    dom_disc_loss =  - tf.reduce_mean(
        tf.reduce_mean(Z_original_domain*tf_log(dom_original) + (-Z_original_domain+1)*tf_log(-dom_original+1), axis=1)
    )
    discriminator_loss = image_disc_loss + dom_disc_loss

    # domain_labels = [
        # "5ClkShadow","ArchEyebrw","Attractive","BagUndrEye","Bald","Bangs","BigLips","BigNose","BlackHair","BlondHair","Blurry","BrownHair","BshyEyebrw","Chubby","DoubleChin","Eyeglasses","Goatee","GrayHair","HvyMakeup","HiCheekbone","Male","MouthSltOpen","Mustache","NarrowEyes","NoBeard","OvalFace","PaleSkin","PointyNose","RecedHairln","RosyCheeks","Sideburns","Smiling","StraightHair","WavyHair","WearEarring","WearHat","WearLipstick","WearNecklc","WearNecktie","Young"
    # ]
    domain_labels = [
        "BlackHair","HvyMakeup","MouthSltOpen","NarrowEyes","NoBeard","Smiling","StraightHair","WearLipstick"
    ]

    def generate_label_image(dom):
        domain_label_image = Image.new("RGB", (WIDTH*1, HEIGHT), (1, 1, 1))
        for i in range(len(domain_labels)):
            label = domain_labels[i]
            draw = ImageDraw.Draw(domain_label_image)
            draw.rectangle((0, i*(HEIGHT//8), WIDTH-1, i*(HEIGHT//8)+HEIGHT//8-1), outline=(0, 0, 0))
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 24)
            draw.multiline_text((2, i*(HEIGHT//8)+2), label, fill=(0, 0, 0), font=font)
        domain_label_image = tf.convert_to_tensor(np.asarray(domain_label_image), dtype=tf.uint8)

        domain = tf.split(dom, ndomains, axis=1)
        domain_image = [tf.tile(tf.reshape(255-d*192, [BATCH_SIZE, 1, 1, 1]), [1, (HEIGHT//8), WIDTH, 1]) for d in domain]
        domain_image = tf.concat(domain_image, axis=1)
        domain_image = tf.concat([domain_image*0+255, domain_image, domain_image], axis=3)
        domain_image = tf.cast(domain_image, tf.float32)
        domain_image = tf.clip_by_value(domain_image*tf.cast(domain_label_image, tf.float32), 0, 255)
        domain_image = tf.cast(domain_image, tf.uint8)
        return domain_image

    target_domain_image = generate_label_image(Z_target_domain)
    target_disc_domain_image = generate_label_image(dom_gen)
    original_domain_image = generate_label_image(Z_original_domain)
    original_disc_domain_image = generate_label_image(dom_original)

    reconst_loss = tf.reduce_mean(tf.abs(original-reconst)) * 10
    identity_loss = tf.reduce_mean(tf.abs(original-identity)) * 3
    ad_image_loss = - tf.reduce_mean(tf_log(p_gen))
    ad_domain_loss = -tf.reduce_mean(Z_target_domain*tf_log(dom_gen) + (-Z_target_domain+1)*tf_log(-dom_gen+1))
    generator_loss = reconst_loss + identity_loss + ad_image_loss + ad_domain_loss

    tf.summary.scalar("generator_loss", generator_loss)
    tf.summary.scalar("discriminator_loss", discriminator_loss)
    tf.summary.scalar("image_disc_loss", image_disc_loss)
    tf.summary.scalar("dom_discr_loss", dom_disc_loss)
    tf.summary.scalar("reconst_loss", reconst_loss)
    tf.summary.scalar("identity_loss", identity_loss)
    tf.summary.scalar("gp_loss", gp_loss)
    tf.summary.scalar("ad_image_loss", ad_image_loss)
    tf.summary.scalar("ad_domain_loss", ad_domain_loss)

    original_image = tf.cast(tf.clip_by_value(original*128.0+128.0, 0, 255), tf.uint8)
    gen_image = tf.cast(tf.clip_by_value(gen*128.0+128.0, 0, 255), tf.uint8)
    reconst_image = tf.cast(tf.clip_by_value(reconst*128.0+128.0, 0, 255), tf.uint8)
    summary_images = tf.concat([original_image, gen_image, reconst_image, original_domain_image, original_disc_domain_image, target_domain_image, target_disc_domain_image], 2)

    tf.summary.image("original/gen/reconst/orig_dom/org_disc_dom/targ_dom/targ_disc_dom", summary_images, BATCH_SIZE)
    summary = tf.summary.merge_all()

    return Z_image, Z_original_domain, Z_target_domain, generator_loss, discriminator_loss, reconst_loss, ad_image_loss, ad_domain_loss, summary

# def grad_scale(loss, variables, scale):
    # vars_grads = [[v, tf.clip_by_value(grad/scale, -0.1, 0.1)] for v, grad in zip(variables, tf.gradients(loss*scale, variables)) if grad is not None]
    # variables = [v[0] for v in vars_grads]
    # grads = [v[1] for v in vars_grads]
    # return variables, grads

# def fp32_get_var(getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, *args, **kwargs):
    # storage_dtype = tf.float32 if trainable else dtype
    # variable = getter(name, shape, dtype=storage_dtype, initializer=initializer, regularizer=regularizer, trainable=trainable, *args, **kwargs)
    # if trainable and dtype != tf.float32:
        # variable = tf.cast(variable, dtype)
    # return variable

# with tf.variable_scope("star_gan", custom_getter=fp32_get_var):
with tf.variable_scope("star_gan"):
    original_tf, original_domain_tf, target_domain_tf, generator_loss_tf, discriminator_loss_tf, reconst_loss_tf, ad_image_loss_tf, ad_domain_loss_tf, summary_tf = setup()

    star_gan_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=".*generator/*")
    star_gan_vars_dict = dict([(var.op.name, var) for var in star_gan_vars])
    star_gan_saver = tf.train.Saver(star_gan_vars_dict, max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=3)

    generator_vars = [x for x in tf.trainable_variables() if "generator" in x.name]
    discriminator_vars = [x for x in tf.trainable_variables() if "discriminator" in x.name]
    learning_rate = args.learning_rate
    generator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, epsilon=1e-8).minimize(generator_loss_tf, var_list=generator_vars)
    # generator_vars_with_grads, generator_grads = grad_scale(generator_loss_tf, generator_vars, 16.0)
    # generator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, epsilon=1e-8)
    # generator_optimizer = generator_optimizer.apply_gradients(zip(generator_grads, generator_vars_with_grads))
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, epsilon=1e-8).minimize(discriminator_loss_tf, var_list=discriminator_vars)
    # discriminator_vars_with_grads, discriminator_grads = grad_scale(discriminator_loss_tf, discriminator_vars, 64.0)
    # discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5, epsilon=1e-8)
    # discriminator_optimizer = discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_vars_with_grads))

    tf.global_variables_initializer().run()

def train(images, n_epochs, batch_size):
    images, domains, image_names = images
    num_images = len(images)
    train_images = np.asarray([images[i] for i in range(num_images) if i%10<8])
    train_image_names = np.asarray([image_names[i] for i in range(num_images) if i%10<8])
    test_images = np.asarray([images[i] for i in range(num_images) if i%10>=8])
    train_domains = np.asarray([domains[i] for i in range(num_images) if i%10<8])
    test_domains = np.asarray([domains[i] for i in range(num_images) if i%10>=8])
    train_log_dir = "log/train"
    test_log_dir = "log/test"
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)
    saver = tf.train.Saver()
    if tf.train.checkpoint_exists("./star_gan.ckpt"):
        saver.restore(sess, "./star_gan.ckpt")
    itr = 0
    best_loss=1e16
    for epoch in range(n_epochs):
        # star_gan_saver.save(sess, "generator_checkpoints/star_gan.ckpt")
        index = np.arange(len(train_images))
        np.random.shuffle(index)
        original_images = train_images[index]
        original_image_names = train_image_names[index]
        original_domains = train_domains[index]

        index = np.arange(len(test_images))
        np.random.shuffle(index)
        original_test_images = test_images[index]
        original_test_domains = test_domains[index]

        generator_loss_total = 0
        discriminator_loss_total = 0
        reconst_loss_total = 0
        ad_image_loss_total = 0
        ad_domain_loss_total = 0
        count = 0
        next_batch = None
        executor = ThreadPoolExecutor(max_workers=2)
        def next_batch(s, e):
            domain = np.random.randint(0, 3, (batch_size, ndomains))
            domain[domain==2] = 0
            target_domain = domain.astype(np.float32)
            original = get_dataset(original_images[s:e])
            return original, target_domain
        next_batch_generator = None

        for start, end in zip(
                range(0, len(original_images), batch_size),
                range(batch_size, len(original_images), batch_size)):
            if next_batch_generator!=None:
                batch = next_batch_generator.result()
                original_batch, target_domain_batch = next_batch_generator.result()
                original_domain_batch = original_domains[start:end]
            else:
                original_batch = original_images[start:end]
                original_batch = get_dataset(original_batch)
                original_domain_batch = original_domains[start:end]
                domain = np.random.randint(0, 2, (batch_size, ndomains))
                domain[domain==2] = 0
                target_domain_batch = domain.astype(np.float32)
            if end+batch_size<len(original_images):
                next_batch_generator = executor.submit(next_batch, start+batch_size, end+batch_size)

            if itr%2<1: # train generator
                _, train_summary_str, generator_loss, discriminator_loss, reconst_loss, ad_image_loss, ad_domain_loss = sess.run([generator_optimizer, summary_tf, generator_loss_tf, discriminator_loss_tf, reconst_loss_tf, ad_image_loss_tf, ad_domain_loss_tf], feed_dict={original_tf: original_batch, original_domain_tf: original_domain_batch, target_domain_tf: target_domain_batch})
            else: # train discriminator
                _, train_summary_str, generator_loss, discriminator_loss, reconst_loss, ad_image_loss, ad_domain_loss = sess.run([discriminator_optimizer, summary_tf, generator_loss_tf, discriminator_loss_tf, reconst_loss_tf, ad_image_loss_tf, ad_domain_loss_tf], feed_dict={original_tf: original_batch, original_domain_tf: original_domain_batch, target_domain_tf: target_domain_batch})

            summary_interval = BATCH_SIZE*16
            if start%summary_interval==0:
                original_test_batch = original_test_images[int(start/summary_interval*batch_size):(int(start/summary_interval*batch_size) + batch_size)]
                original_test_batch = get_dataset(original_test_batch)
                original_domain_test_batch = original_test_domains[int(start/summary_interval*batch_size):(int(start/summary_interval*batch_size) + batch_size)]
                domain = np.random.randint(0, 2, (batch_size, ndomains))
                domain[domain==2] = 0
                target_domain_batch = domain.astype(np.float32)
                if original_domain_test_batch.size==0:
                    continue
                test_summary, generator_loss, discriminator_loss, reconst_loss, ad_image_loss, ad_domain_loss = sess.run([summary_tf, generator_loss_tf, discriminator_loss_tf, reconst_loss_tf, ad_image_loss_tf, ad_domain_loss_tf], feed_dict={original_tf: original_test_batch, original_domain_tf: original_domain_test_batch, target_domain_tf: target_domain_batch})
                test_summary_writer.add_summary(test_summary, itr)
                train_summary_writer.add_summary(train_summary_str, itr)

            count += 1
            generator_loss_total += generator_loss
            discriminator_loss_total += discriminator_loss if discriminator_loss!=inf else (discriminator_loss_total/count)
            reconst_loss_total += reconst_loss
            ad_image_loss_total += ad_image_loss
            ad_domain_loss_total += ad_domain_loss

            train_summary_writer.flush()
            test_summary_writer.flush()
            print("epoch {epc}, {cur}/{total}: gen_loss={gen_loss:.5f}({gen_loss_avg:.5f}), disc_loss={disc_loss:.5f}({disc_loss_avg:.5f}), reconst_loss={reconst_loss:.5f}({reconst_loss_avg:.5f}), ad_image_loss={ad_image_loss:.5f}({ad_image_loss_avg:.5f}), ad_domain_loss={ad_domain_loss:.5f}({ad_domain_loss_avg:.5f})".format(
                epc=epoch, cur=start, total=len(train_images), gen_loss=generator_loss, gen_loss_avg=generator_loss_total/count, disc_loss=discriminator_loss, disc_loss_avg=discriminator_loss_total/count,
                reconst_loss=reconst_loss, reconst_loss_avg=reconst_loss_total/count, ad_image_loss=ad_image_loss, ad_image_loss_avg=ad_image_loss_total/count, ad_domain_loss=ad_domain_loss, ad_domain_loss_avg=ad_domain_loss_total/count)
            )
            itr += 1
        avg_loss = generator_loss_total/count
        if avg_loss<best_loss:
            best_loss = avg_loss
            if epoch>=1:
                saver.save(sess, "./star_gan.ckpt")
        saver.save(sess, "latest/star_gan.ckpt")
        cost_log = open("./cost.log", mode="a")
        cost_log.write("epoch=%d, gen_loss=%f, disc_loss=%f\n" % (epoch, generator_loss_total/count, discriminator_loss_total/count))
        cost_log.close()
    train_summary_writer.close()
    test_summary_writer.close()
train(dataset, nepochs, BATCH_SIZE)
