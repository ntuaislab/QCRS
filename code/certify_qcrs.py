# evaluate a smoothed classifier on a dataset
import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from core_qcrs import Smooth 
from time import time
import torch
import datetime
from architectures import get_architecture

from ipdb import set_trace as st
import numpy as np
import random
from macer_models import resnet110


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--sigma", default=0.12, type=float, help="noise hyperparameter")
parser.add_argument("--dataset", default='cifar10', choices=DATASETS, help="which dataset (cifar10/imagenet)")
parser.add_argument("--batch", type=int, default=50000, help="batch size")
parser.add_argument("--skip", type=int, default=5, help="how many examples to skip, total 10000")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--epsilon", type=float, default=0.001, help="to achieve epsilon siboptimal")
parser.add_argument("--left", type=float, default=0.07, help="left (min) sigma")
parser.add_argument("--right", type=float, default=0.36, help="right (max) sigma")
parser.add_argument("--epsilon_step", type=float, default=2.0, help="step (to compute gradient) = eplison/epsilon_step")
parser.add_argument("--gradient_n", type=int, default=500, help="the samples that used to compute the fradient")
parser.add_argument('--odd_imagenet', action='store_true', help='options: if setup, run X%skip==100. OW, X%skip==0')
parser.add_argument('--macer', action='store_true', help='if True, load macer weight')
parser.add_argument('--cohen', action='store_true', help='if True, use cohen instead of the proposed method')

args = parser.parse_args()
os.makedirs(f'./exp/{args.dataset}', exist_ok=True)
args.outfile = os.path.join(f'./exp/{args.dataset}', args.outfile)

# if args.dataset == 'imagenet':
#     if args.skip < 100:
#         args.skip = 100
#     args.batch = 1000

seed = 888
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    checkpoint = torch.load(args.base_classifier)
    if args.macer:
        base_classifier = resnet110()
        base_classifier.load_state_dict(checkpoint['net'])
        base_classifier = base_classifier.cuda()
    else:
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset))
    base_classifier.eval()
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tbest_sigma\tpABar\ttime", file=f, flush=True)

    dataset = get_dataset(args.dataset, args.split)
    
    if args.odd_imagenet == True and args.dataset=='imagenet': 
        skip_factor=100 
    else: 
        skip_factor=0

    curve_collection = []
    for i in range(len(dataset)):
        if i % args.skip != skip_factor:
            continue
        if i == args.max:
            break
        (x, label) = dataset[i]
        before_time = time()
        x = x.cuda()
        if args.cohen:
            best_sigma = args.sigma
        else:
            best_sigma = smoothed_classifier.binary_search_bisection(x, n=args.gradient_n, epsilon=args.epsilon, left_sigma=args.left, right_sigma=args.right, 
                                                                original_sigma=args.sigma, epsilon_step=args.epsilon_step)
        prediction, radius, pABar = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, best_sigma)
        after_time = time()
        correct = int(prediction == label)
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print(f'{i}\t{label}\t{prediction}\t{radius}\t{correct}\t{best_sigma}\t{pABar}\t{time_elapsed}', file=f, flush=True)
        print(f'{i}/{len(dataset)} | {(i)/len(dataset)*100:.2f}% | Remaining: {datetime.timedelta(seconds=round(((len(dataset)-i)/args.skip)*(after_time - before_time)))} | Output: {args.outfile}', end='\r')
    print()


