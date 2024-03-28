# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
import torch.nn as nn

from torch.autograd import Variable
from torch.distributions.normal import Normal
from ipdb import set_trace as st
import copy
from macer_models import resnet110


def OptimzeSigma(model, batch, alpha=1e-4, sig_0=0.5, K=100, n=8):
    '''from DDRS'''
    device='cuda:0'
    batch = batch.unsqueeze(0)
    batch_size = batch.shape[0]
    sig = Variable(torch.tensor(sig_0), requires_grad=True).view(batch_size, 1, 1, 1).to(device)
    m = Normal(torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device))

    # for param in model.parameters():
        # param.requires_grad_(False)

    #Reshaping so for n > 1
    new_shape = [batch_size * n]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1,n, 1, 1)).view(new_shape)
    new_batch = new_batch.to(device)

    for _ in range(K):
        sigma_repeated = sig.repeat((1, n, 1, 1)).view(-1,1,1,1).to(device)
        eps = torch.randn_like(new_batch)*sigma_repeated #Reparamitrization trick
        out = model(new_batch + eps).reshape(batch_size, n, -1).mean(1)#This is \psi in the algorithm
        out = torch.nn.functional.softmax(out, dim=1)
        
        vals, _ = torch.topk(out, 2)
        vals.transpose_(0, 1)
        gap = m.icdf(vals[0].clamp_(0.02, 0.98)) - m.icdf(vals[1].clamp_(0.02, 0.98))
        radius = sig.reshape(-1)/2 * gap  # The radius formula
        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += alpha*grad[0]  # Gradient Ascent step
    #For training purposes after getting the sigma
    # for param in model.parameters():
        # param.requires_grad_(True)    

    return sig.reshape(-1)



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("path", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=50000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--kk", type=int, default=100, help="optimize step for sigme")
parser.add_argument("--nn", type=int, default=1, help="number to estimate expection")
parser.add_argument('--macer', action='store_true', help='if True, load macer weight')
# parser.add_argument('--fix-sig-smooth', action='store_true', default=False, help='certify with fixed sigma')
# parser.add_argument("--path_sigma", type=str, help="path to sigma")
args = parser.parse_args()

print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def get_model(path):
#     model = resnet.resnet18(num_classes=10).to('cuda')
#     checkpoint = torch.load(path, map_location='cuda')
#     model.load_state_dict(checkpoint['state_dict'])
#     print('Pretrained Model is loaded ! Go and certify now :)')
#     return model


if __name__ == "__main__":
    checkpoint = torch.load(args.path)
    
    if args.macer:
        base_classifier = resnet110()
        base_classifier.load_state_dict(checkpoint['net'])
        base_classifier = base_classifier.cuda()
    else:
        base_classifier = get_architecture(checkpoint["arch"], args.dataset)
        base_classifier.load_state_dict(checkpoint['state_dict'])
    
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\tsigma\ttime", file=f, flush=True)
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset))
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        before_time = time()
        (x, label) = dataset[i]
       
        x = x.cuda()

        best_sigma = OptimzeSigma(copy.deepcopy(base_classifier), batch=x.clone(), alpha=1e-4, sig_0=0.5, K=100, n=1).item()

        print(f'#{i} sigma is: ', best_sigma)
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch, best_sigma)
        print(best_sigma, prediction, radius)

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{:.3}\t{}".format(
            i, label, prediction, radius, correct, best_sigma, time_elapsed), file=f, flush=True)
        print(f'{i}/{len(dataset)} | {(i)/len(dataset)*100:.2f}% | Remaining: {datetime.timedelta(seconds=round(((len(dataset)-i)/args.skip)*(after_time - before_time)))} | Output: {args.outfile}')

    f.close()
