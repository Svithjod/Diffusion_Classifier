import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from model import DDPM, ContextUnet

def get_args_parser():
    parser = argparse.ArgumentParser('Guided Diffusion', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=20, type=int)

    # Model parameters
    parser.add_argument('--n_feat', default=256, type=int)
    parser.add_argument('--n_T', default=400, type=int)

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--n_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./out/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_model', action='store_true')

    return parser



def train_mnist(args):

    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=args.n_feat, n_classes=args.n_classes), 
                n_classes=args.n_classes, betas=(1e-4, 0.02), n_T=args.n_T, device=args.device, drop_prob=0.1)
    ddpm.to(args.device)

    print('Model = %s' % str(ddpm))

    # optionally load a model
    # ddpm.load_state_dict(torch.load('./data/diffusion_outputs/ddpm_unet01_mnist_9.pth'))

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MNIST(args.data_path, train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    optim = torch.optim.Adam(ddpm.parameters(), lr=args.lr)

    eval_freq = 5
    for ep in range(args.start_epoch, args.epochs):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = args.lr*(1-ep/args.epochs)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(args.device)
            c = c.to(args.device)
            loss = ddpm.loss(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f'loss: {loss_ema:.4f}')
            optim.step()


        if args.eval and (ep%eval_freq==0 or ep == int(args.epochs-1)):
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            ddpm.eval()
            ws_test = [0.0, 2.0] # strength of generative guidance
            with torch.no_grad():
                n_sample = 4*args.n_classes
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), args.device, guide_w=w)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(args.device)
                    for k in range(args.n_classes):
                        for j in range(int(n_sample/args.n_classes)):
                            try: 
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k+(j*args.n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all*-1 + 1, nrow=10)
                    image_path = os.path.join(args.output_dir, f'image_ep{ep}_w{w}.png')
                    save_image(grid, image_path)
                    print('saved image at ' + image_path)

                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample/args.n_classes), ncols=args.n_classes,
                                            sharex=True, sharey=True, figsize=(8,3))
                    def animate_diff(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample/args.n_classes)):
                            for col in range(args.n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(-x_gen_store[i,(row*args.n_classes)+col,0],
                                                                  cmap='gray',
                                                                  vmin=(-x_gen_store[i]).min(),
                                                                  vmax=(-x_gen_store[i]).max()))
                        return plots
                    ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],
                                        interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
                    gif_path = os.path.join(args.output_dir, f'gif_ep{ep}_w{w}.gif')
                    ani.save(gif_path, dpi=100, writer=PillowWriter(fps=5))
                    print('saved image at ' + gif_path)

        # optionally save model
        if args.save_model and ep == int(args.epochs-1):
            model_path = os.path.join(args.output_dir, f'model_{ep}.pth')
            torch.save(ddpm.state_dict(), model_path)
            print('saved model at ' + model_path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train_mnist(args)

