import random
random.seed(7)
import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .data_reader import H5DataLoader
from .img_utils import imsave
from .denseunet import DenseUnet
from .acmdenseunet import AcmDenseUnet
from .ccv import CCV
from src.DACN import ops

class Actions(object):
#—————————————————————————————————————————————————————#
    def __init__(self, sess, conf):

        #——————————————  step：1  ——————————————#
        print("Actions 1", flush=True)
        self.sess = sess
        self.conf = conf
        self.conv_size = (3, 3)
        self.pool_size = (2, 2)
        self.data_format = 'NHWC'
        self.axis, self.channel_axis, self.batch_axis = (1, 2), 3, 0

        self.input_shape = [conf.batchsize, conf.height, conf.width, conf.channel]
        self.output_shape = [conf.batchsize, conf.height, conf.width]

        #——————————————  step：2  ——————————————#
        print("Actions 2", flush=True)
        
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
        if not os.path.exists(conf.record_dir):
            os.makedirs(conf.record_dir)

        #——————————————  step：3  ——————————————#
        print("Actions 3", flush=True)
        
        if self.conf.gpu_num==1:
            self.configure_networks_single()
        else:
            self.configure_networks_multi()
            
        print("Actions 4", flush=True)
            
#———————————————————————————— configure_networks_single —————————————————————————#
    def configure_networks_single(self):

        print("CNS 1", flush=True)

        self.inputs = tf.placeholder(
            tf.float32,
            self.input_shape,
            name='inputs'
        )

        self.is_train = tf.placeholder(
            tf.bool,
            name='is_train'
        )

        print("CNS 2", flush=True)

        if self.conf.network_name == "denseunet":
            model = DenseUnet(self.sess, self.conf, self.is_train)
            self.outputs, self.rates = model.inference(self.inputs)

        elif self.conf.network_name == "acmdenseunet":
            model = AcmDenseUnet(self.sess, self.conf, self.is_train)
            self.outputs, self.rates = model.inference(self.inputs)

        print("CNS 3", flush=True)

        if self.conf.network_name in ["unet", "denseunet"]:

            self.pred = self.outputs

        else:

            self.net_pred = self.outputs[:, :, :, 2:]

            self.pred = CCV(
                self.outputs,
                self.inputs,
                2,
                0.5,
                1e-8
            )

            self.pred = tf.squeeze(self.pred)

        print("CNS 4", flush=True)

        gamma = 0.5

        high0 = tf.ones(tf.shape(self.pred), dtype=tf.int64)
        low0 = tf.zeros(tf.shape(self.pred), dtype=tf.int64)

        gamma0 = tf.ones(tf.shape(self.pred)) * gamma

        self.decoded_predictions = tf.where(
            tf.greater_equal(self.pred, gamma0),
            high0,
            low0
        )

        print("CNS 5", flush=True)

        tf.set_random_seed(self.conf.random_seed)

        trainable_vars = tf.trainable_variables()

        g_list = tf.global_variables()

        bn_moving_vars = [
            g for g in g_list
            if 'batch_norm/moving_mean' in g.name
        ]

        bn_moving_vars += [
            g for g in g_list
            if 'batch_norm/moving_variance' in g.name
        ]

        trainable_vars += bn_moving_vars

        self.saver = tf.train.Saver(
            var_list=trainable_vars,
            max_to_keep=0
        )

        print("CNS 6", flush=True)
#———————————————————————————— train —————————————————————————#
    def train(self):


        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')

        if self.conf.reload_epoch > 0:
            self.reload(self.conf.reload_epoch)

        train_reader = H5DataLoader(self.conf.data_dir+self.conf.train_data)
        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data)

        valid_loss_list = []
        train_loss_list = []

        train_acc_list = []
        valid_acc_list = []

        train_miou_list = []
        valid_miou_list = []

        self.sess.run(tf.local_variables_initializer())

        for epoch_num in range(self.conf.max_epoch):


            if epoch_num % self.conf.test_step == 1:
                inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
                if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, summary = self.sess.run([self.loss_op, self.valid_summary], feed_dict=feed_dict)
                loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)

                print(epoch_num, '----valid loss', loss)

                # loss
                valid_loss_list.append(loss)
                np.save(self.conf.record_dir+"valid_loss.npy",np.array(valid_loss_list))
                # acc
                valid_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"valid_acc.npy",np.array(valid_acc_list))
                # miou
                valid_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"valid_miou.npy",np.array(valid_miou_list))

                #########################################################################
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                haha, loss, accuracy, m_iou, _ = self.sess.run([self.train_op, self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)

                print(epoch_num, '----train loss', loss)

                # loss
                train_loss_list.append(loss)
                np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
                # acc
                train_acc_list.append(accuracy)
                np.save(self.conf.record_dir+"train_acc.npy",np.array(train_acc_list))
                # miou
                train_miou_list.append(m_iou)
                np.save(self.conf.record_dir+"train_miou.npy",np.array(train_miou_list))

            elif epoch_num % self.conf.summary_step == 1:
                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
                #loss, _, summary = self.sess.run([self.loss_op, self.train_op, self.train_summary], feed_dict=feed_dict)
                #self.save_summary(summary, epoch_num)
                #print(epoch_num)


                #train_loss_list.append(loss)
                #np.save(self.conf.record_dir+"train_loss.npy",np.array(train_loss_list))
            else:

                inputs, annotations = train_reader.next_batch(self.conf.batchsize)
                if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

                feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: True}
                loss,_ = self.sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)

                print(epoch_num)


            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)
#———————————————————————————— test —————————————————————————#

    def test(self,model_i):

        print('---->testing ', model_i)

        if model_i > 0:
            self.reload(model_i)
        else:
            print("please set a reasonable test_epoch")
            return


        valid_reader = H5DataLoader(self.conf.data_dir+self.conf.valid_data,False)
        self.sess.run(tf.local_variables_initializer())

        losses = []
        accuracies = []
        m_ious = []
        dices = []
        count = 0
        while True:
            inputs, annotations = valid_reader.next_batch(self.conf.batchsize)
            if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

            if inputs.shape[0] < self.conf.batch:
                break

            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _ = self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print(count)
            print('values----->', loss, accuracy, m_iou)
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)

            out, gt = self.sess.run([self.out, self.gt], feed_dict=feed_dict)

            if self.conf.class_num==2:
                tp = np.sum(out*gt)
                fenmu = np.sum(out)+np.sum(gt)+0.000001
                dice = 2*tp/fenmu
                dices.append(dice)

            print('dice----->', dice)
            count+=1
            if count==self.conf.valid_num:
                break

        return np.mean(losses),np.mean(accuracies),m_ious[-1],np.mean(dices)
#———————————————————————————— predict —————————————————————————#

    def predict(self):

        print('---->predicting ', self.conf.test_epoch)

        if self.conf.test_epoch > 0:
            self.reload(self.conf.test_epoch)
        else:
            print("please set a reasonable test_epoch")
            return

        test_reader = H5DataLoader(self.conf.data_dir+self.conf.test_data, False)
        self.sess.run(tf.local_variables_initializer())
        predictions = []
        net_predictions = []
        outputs = []
        probabilitys = []
        losses = []
        accuracies = []
        m_ious = []

        rate_list = []
        befores = []
        afters = []
        maps = []
        start_maps = []
        count = 0

        while True:
            inputs, annotations = test_reader.next_batch(self.conf.batchsize)
            if annotations.ndim == 4 and annotations.shape[-1] == 1:
                  annotations = annotations[..., 0]

            if inputs.shape[0] < self.conf.batch:
                break

            feed_dict = {self.inputs: inputs, self.annotations: annotations, self.is_train: False}
            loss, accuracy, m_iou, _= self.sess.run([self.loss_op, self.accuracy_op, self.m_iou, self.miou_op], feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)

            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)

            predictions.append(self.sess.run(self.decoded_predictions, feed_dict=feed_dict))
            net_predictions.append(self.sess.run(self.decoded_net_pred, feed_dict=feed_dict))
            outputs.append(self.sess.run(self.outputs, feed_dict=feed_dict))

            count+=1
            if count==self.conf.test_num:
                break

        print('----->saving outputs')
        print(np.shape(probabilitys))
        np.save(self.conf.sample_dir+"outputs"+".npy",np.array(outputs))

        print('----->saving predictions')
        print(np.shape(predictions))
        num = 0
        for index, prediction in enumerate(predictions):
            for i in range(prediction.shape[0]):
                np.save(self.conf.sample_dir+"pred"+str(num)+".npy", prediction[i])
                num += 1
                # gunakan imsave fixed
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i) + '.png')

        print('----->saving net_predictions')
        print(np.shape(net_predictions))
        num = 0
        for index, prediction in enumerate(net_predictions):
            for i in range(prediction.shape[0]):
                np.save(self.conf.sample_dir+"netpred"+str(num)+".npy", prediction[i])
                num += 1
                imsave(prediction[i], self.conf.sample_dir + str(index*prediction.shape[0]+i) + 'net.png')

        return np.mean(losses),np.mean(accuracies),m_ious[-1]
#———————————————————————————— config_summary —————————————————————————#

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.image(name+'/input', self.inputs, max_outputs=100))
        summarys.append(tf.summary.image(name + '/annotation', tf.cast(tf.expand_dims(
                self.annotations, -1), tf.float32), max_outputs=100))
        summarys.append(tf.summary.image(name + '/prediction', tf.cast(tf.expand_dims(
                self.decoded_predictions, -1), tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary
#———————————————————————————— save_summary —————————————————————————#

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)
#———————————————————————————— save —————————————————————————#

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)
#———————————————————————————— reload —————————————————————————#

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
#———————————————————————————— predict_image —————————————————————————#

    def predict_image(self, image):
        """
        image:
            numpy array shape (H,W,3)

        return:
            mask shape (256,256)
        """

        image = image.astype(np.float32)

        if image.shape[:2] != (self.conf.height, self.conf.width):
            import cv2
            image = cv2.resize(
                image,
                (self.conf.width, self.conf.height)
            )

        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, self.conf.batchsize, axis=0)
        
        feed_dict = {
            self.inputs: image,
            self.is_train: False
        }

        prediction = self.sess.run(
            self.decoded_predictions,
            feed_dict=feed_dict
        )

        return prediction[0]