import os
import time
import argparse
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from actions import Actions


def configure():
    flags = tf.app.flags

    #————————————————————————————--—————————————————————————#
    flags.DEFINE_string('network_name', 'acmdenseunet', 'Use which framework:  unet, denseunet, deeplabv3plus')

    # flags.DEFINE_integer('max_epoch', 30000, '# of step in an epoch')
    # flags.DEFINE_integer('test_step', 500, '# of step to test a model')
    # flags.DEFINE_integer('save_step', 500, '# of step to save a model')
    flags.DEFINE_integer('max_epoch', 1000, '# of step in an epoch')
    flags.DEFINE_integer('test_step', 50, '# of step to test a model')
    flags.DEFINE_integer('save_step', 50, '# of step to save a model')

    flags.DEFINE_integer('valid_start_epoch',1,'start step to test a model')
    # flags.DEFINE_integer('valid_end_epoch',30001,'end step to test a model')
    # flags.DEFINE_integer('valid_stride_of_epoch',500,'stride to test a model')
    flags.DEFINE_integer('valid_end_epoch',1001,'end step to test a model')
    flags.DEFINE_integer('valid_stride_of_epoch',50,'stride to test a model')
    flags.DEFINE_string('model_name', 'model', 'Model file name')
    flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
    # flags.DEFINE_integer('test_epoch', 26501, 'Test or predict epoch')
    flags.DEFINE_integer('test_epoch', 1, 'Test or predict epoch')
    flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')

    # flags.DEFINE_integer('summary_step', 10000000, '# of step to save the summary')
    flags.DEFINE_integer('summary_step', 50, '# of step to save the summary')
    #—————————————————————————————————————————————————————#

    # flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
    flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
    flags.DEFINE_float('beta1', 0.9, 'beta1')
    flags.DEFINE_float('beta2', 0.99, 'beta2')
    flags.DEFINE_float('epsilon', 1e-8, 'epsilon')

    flags.DEFINE_integer('gpu_num', 1, 'the number of GPU')
    #—————————————————————————————————————————————————————#
    flags.DEFINE_string('data_dir', '/content/herlev_upscaled_h5/', 'Name of data directory')
    flags.DEFINE_string('train_data', 'herlev_train.h5', 'Training data')
    flags.DEFINE_string('valid_data', 'herlev_valid.h5', 'Validation data')
    flags.DEFINE_string('test_data', 'herlev_test.h5', 'Testing data')
    flags.DEFINE_integer('valid_num',184,'the number of images in the validing set')
    flags.DEFINE_integer('test_num',92,'the number of images in the testing set')
    flags.DEFINE_integer('batch', 4, 'batch size')
    flags.DEFINE_integer('batchsize', 4, 'total batch size')
    flags.DEFINE_integer('channel', 3, 'channel size')
    flags.DEFINE_integer('height', 256, 'height size')
    flags.DEFINE_integer('width', 256, 'width size')
    #flags.DEFINE_integer('cropsize', 128, 'crop size') #
    flags.DEFINE_boolean('is_training', True, '是否训练')
    flags.DEFINE_integer('class_num', 2, 'output class number')
    #————————————————————————————-—————————————————————————#
    flags.DEFINE_string('logdir', '/content/herlev_original_upscaled_ep1000_lr1e4/logdir', 'Log dir')
    flags.DEFINE_string('modeldir', '/content/herlev_original_upscaled_ep1000_lr1e4/modeldir', 'Model dir')
    flags.DEFINE_string('sample_dir', '/content/herlev_original_upscaled_ep1000_lr1e4/samples/', 'Sample directory')
    flags.DEFINE_string('record_dir', '/content/herlev_original_upscaled_ep1000_lr1e4/record/', 'Experiment record directory')
    #————————————————————————————-—————————————————————————#
    flags.DEFINE_boolean('use_asc', False, 'use ASC or not')
    flags.DEFINE_string('down_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d, adaptive_separate_conv2d')
    flags.DEFINE_string('bottom_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')
    flags.DEFINE_string('up_conv_name', 'conv2d', 'Use which conv op: conv2d, deform_conv2d, adaptive_conv2d')

    flags.DEFINE_string('deconv_name', 'deconv', 'Use which deconv op: deconv')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS
#———————————————————————————— train —————————————————————————#
"""
函数功能：训练
"""
def train():
    model = Actions(sess, configure())
    model.train()
#———————————————————————————— valid —————————————————————————#
"""
函数功能：验证
"""
def valid():
    valid_loss = []
    valid_accuracy = []
    valid_m_iou = []
    valid_dice =[]
    valid_epochs = []

    conf = configure()
    model = Actions(sess, conf)
    for i in range(conf.valid_start_epoch,conf.valid_end_epoch,conf.valid_stride_of_epoch):
        try:
            loss, acc, m_iou, dice = model.test(i)
        except Exception as e:
            print("Skip checkpoint", i, e)
            continue

        valid_epochs.append(i)
        valid_loss.append(loss)
        valid_accuracy.append(acc)
        valid_m_iou.append(m_iou)
        valid_dice.append(dice)
        np.save(conf.record_dir+"validate_epoch.npy",np.array(valid_epochs))
        np.save(conf.record_dir+"validate_loss.npy",np.array(valid_loss))
        np.save(conf.record_dir+"validate_accuracy.npy",np.array(valid_accuracy))
        np.save(conf.record_dir+"validate_m_iou.npy",np.array(valid_m_iou))
        np.save(conf.record_dir+"validate_dice.npy",np.array(valid_dice))
        print('valid_loss',valid_loss)
        print('valid_accuracy',valid_accuracy)
        print('valid_m_iou',valid_m_iou)
        print('valid_dice',valid_dice)
        print('valid_epoch',valid_epochs)
    
    best_idx = np.argmax(valid_dice)
    best_epoch = valid_epochs[best_idx]

    with open(conf.record_dir + "best_model.txt", "w") as f:
        f.write("Best Epoch : {}\n".format(best_epoch))
        f.write("Best Dice : {}\n".format(valid_dice[best_idx]))
        f.write("Best Accuracy : {}\n".format(valid_accuracy[best_idx]))
        f.write("Best mIoU : {}\n".format(valid_m_iou[best_idx]))
        f.write("Best Loss : {}\n".format(valid_loss[best_idx]))

    print("\n===== BEST MODEL =====")
    print("Epoch    :", best_epoch)
    print("Dice     :", valid_dice[best_idx])
    print("Accuracy :", valid_accuracy[best_idx])
    print("mIoU     :", valid_m_iou[best_idx])
    print("Loss     :", valid_loss[best_idx])
    print("======================")
#———————————————————————————— predict —————————————————————————#
"""
函数功能：测试
"""
def predict():
    predict_loss = []
    predict_accuracy = []
    predict_m_iou = []
    model = Actions(sess, configure())
    loss,acc,m_iou = model.predict()
    predict_loss.append(loss)
    predict_accuracy.append(acc)
    predict_m_iou.append(m_iou)
    print('predict_loss',predict_loss)
    print('predict_accuracy',predict_accuracy)
    print('predict_m_iou',predict_m_iou)
#———————————————————————————— main —————————————————————————#
"""
函数功能：主函数，设置不同的action
"""
def main(argv):
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', dest='action', type=str, default='train',
                        help='actions: train, test, or predict')
    args = parser.parse_args()
    if args.action not in ['train', 'test', 'predict']:
        print('invalid action: ', args.action)
        print("Please input a action: train, test, or predict")
    # test
    elif args.action == 'test':
        valid()
    # predict
    elif args.action == 'predict':
        predict()
    # train
    else:
        train()
    end = time.perf_counter()
    print("program total running time",(end-start)/60)
#———————————————————————————— GPU设置 —————————————————————————#
if __name__ == '__main__':


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.app.run()
