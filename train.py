import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
from model import Pix2Pix

''' Parameters '''
data_dir = "./refocus"
batch_size = 4
image_size = [512, 512]
l1_weight = 10
gp_weight = 1

learn_rate = 1e-2
max_iter = 1000
model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

''' Load data '''
with tf.variable_scope("data_loader", reuse=tf.AUTO_REUSE):
    data = Dataset(data_dir, image_size)
    data_loader = data.load_data_batch().repeat().batch(batch_size) # repeat(epoch) or indefinitely
    pairs_batch = data_loader.make_one_shot_iterator().get_next()

    defocus_batch = pairs_batch[0]
    focus_batch = pairs_batch[1]

''' Build graph'''
with tf.variable_scope("input", reuse=tf.AUTO_REUSE):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    defocus = tf.placeholder(tf.float32, [None] + image_size + [3], name='defocus')
    focus = tf.placeholder(tf.float32, [None] + image_size + [3], name='focus')

with tf.variable_scope("pix2pix" ,reuse=tf.AUTO_REUSE):
    model = Pix2Pix(hierarchy=4, filter_size=32, kernel_size=[5,5], conv_strides=2,
                    batch_size=batch_size, image_size=image_size)

    fake_focus = model.generator(defocus, is_training)
    fake_score = model.discriminator(defocus, fake_focus, is_training)
    true_score = model.discriminator(defocus, focus, is_training)

    # GAN Loss
    g_loss = -tf.reduce_mean(fake_score)
    d_loss = -tf.reduce_mean(true_score) + tf.reduce_mean(fake_score)
    # L1 Loss
    l1_loss = l1_weight * tf.reduce_mean(tf.abs(focus-fake_focus))
    # Gradient Penalty
    gp_loss = gp_weight * model.gradient_penalty(defocus, fake_focus, focus)

    # add names
    fake_focus = tf.identity(fake_focus, name='fake_focus')
    fake_score = tf.identity(fake_score, name='fake_score')
    true_score = tf.identity(true_score, name='true_score')
    g_loss = tf.identity(g_loss, name='g_loss')
    d_loss = tf.identity(d_loss, name='d_loss')
    l1_loss = tf.identity(l1_loss, name='l1_loss')
    gp_loss = tf.identity(gp_loss, name='gp_loss')

''' Train op '''
with tf.variable_scope("training", reuse=tf.AUTO_REUSE):
    global_step = tf.get_default_graph().get_tensor_by_name('input/global_step:0')

    # important to seperate
    g_var = [var for var in tf.trainable_variables() if 'generator' in var.name]
    d_var = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(g_loss+l1_loss, var_list=g_var)
        d_op = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(d_loss+gp_loss, var_list=d_var, global_step=global_step)

''' Summary '''
with tf.variable_scope("summary", reuse=tf.AUTO_REUSE):
    loss_summary = tf.summary.merge([tf.summary.scalar('g_loss', g_loss),
                                     tf.summary.scalar('d_loss', d_loss),
                                     tf.summary.scalar('l1_loss', l1_loss),
                                     tf.summary.scalar('gp_loss', gp_loss)])

    ph_defocus = tf.placeholder(tf.float32, shape=[None,None,None,3])
    ph_focus = tf.placeholder(tf.float32, shape=[None,None,None,3])
    ph_refocus = tf.placeholder(tf.float32, shape=[None,None,None,3])
    image_summary = tf.summary.merge([tf.summary.image('defocus', ph_defocus),
                                      tf.summary.image('focus', ph_focus),
                                      tf.summary.image('refocus', ph_refocus)])

    writer = tf.summary.FileWriter(model_dir)

''' Run session'''
# use GPU memory based on runtime allocation and visible device
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    saver = tf.train.Saver()
    latest_ckpt = tf.train.latest_checkpoint(model_dir)

    if latest_ckpt is not None:
        print('Restoring from ' + latest_ckpt)
        saver.restore(sess, latest_ckpt)
    else:
        print('Starting with a new model')
        tf.train.export_meta_graph(os.path.join(model_dir, 'model.meta')) # save meta for restore

    for j in range(max_iter):
        i = tf.train.global_step(sess, global_step)
        print('iteration: ' + str(j) + ' global step: ' + str(i))

        # load image pairs
        defocus_images, focus_images = sess.run([defocus_batch, focus_batch])

        # normal training
        _, summary = sess.run([d_op, loss_summary], {defocus: defocus_images, focus: focus_images, is_training: True})
        _ = sess.run(g_op, {defocus: defocus_images, focus: focus_images, is_training: True})
        writer.add_summary(summary, i)

        if (i+1)% 200 == 0:
            # visualize
            refocus_images = sess.run(fake_focus, {defocus:defocus_images, is_training: True})
            summary = sess.run(image_summary, {ph_defocus: defocus_images,
                                               ph_focus: focus_images,
                                               ph_refocus: refocus_images})
            writer.add_summary(summary, i)

            # save the model
            saver.save(sess, os.path.join(model_dir, 'model.ckpt'), global_step=i, write_meta_graph=False)