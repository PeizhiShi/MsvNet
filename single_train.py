
import libs.trainer as ltr
import argparse



#this setting produces the sub-optimal results for single-feature recognition
parser = argparse.ArgumentParser(description='Feature Recognition Training')
parser.add_argument('--resolution', dest='resolution', default = 64, action='store_true', help='model resolution: 16, 32, 64')
parser.add_argument('--num_train', dest='num_train', default = 32, action='store_true', help='number of training examples per class')
parser.add_argument('--num_val_test', dest='num_val_test', default = 600, action='store_true', help='number of val/test examples per class')


parser.add_argument('--learning_rate', dest='learning_rate', default = 0.0001, action='store_true', help='learning rate')
parser.add_argument('--epoch1', dest='epoch1', default = 20, action='store_true', help='num of epochs at stage 1')
parser.add_argument('--epoch2', dest='epoch2', default = 100, action='store_true', help='num of epochs at stage 2')
parser.add_argument('--batch_size', dest='batch_size', default = 64, action='store_true', help='batch size')
parser.add_argument('--num_cuts', dest='num_cuts', default = 12, action='store_true', help='number of cuts')
parser.add_argument('--pretrained', dest='pretrained', default = True, action='store_true', help='indicate whether to load pretrained model')
parser.add_argument('--finetuned', dest='finetuned', default = True, action='store_true', help='finetuning')


args = parser.parse_args()
ltr.train_test_model(args)
