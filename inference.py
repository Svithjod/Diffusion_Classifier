from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from model import DDPM, ContextUnet

def get_args_parser():
    parser = argparse.ArgumentParser('Guided Diffusion', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)

    # Model parameters
    parser.add_argument('--n_feat', default=256, type=int)
    parser.add_argument('--n_T', default=400, type=int)
    parser.add_argument('--saved_model', default='', type=str,
                        help='checkpoint path')
    parser.add_argument('--mc_sample', default=10, type=int,
                        help='samples for monte carlo estimate')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--n_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=6, type=int)

    return parser


def inference_mnist(args):

    # don't drop context for inference
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes), 
                n_classes=args.n_classes, betas=(1e-4, 0.02), n_T=args.n_T, device=args.device, drop_prob=0.0)
    ddpm.load_state_dict(torch.load(args.saved_model))
    ddpm.to(args.device)
    ddpm = torch.compile(ddpm)

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST(args.data_path, train=False, download=False, transform=tf)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ddpm.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for x, c in tqdm(dataloader):
            x = x.to(args.device)
            c = c.to(args.device)

            pred = ddpm.inference(x, mc_sample=args.mc_sample)
            total += c.size(0)
            correct += (pred == c).sum().item()

        acc = correct / total
        print(f'Acc on test set: {acc*100:.2f}%')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    inference_mnist(args)

