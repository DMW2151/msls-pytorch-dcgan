# Copyright (C) 2021 Habana Labs, Ltd. an Intel Company

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os

from msls.gan import (
    Discriminator128,
    Generator128,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7744, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        msg = 'Not using distributed mode'
        try:
            from mpi4py import MPI
            global mpi_comm
            mpi_comm = MPI.COMM_WORLD
            size = mpi_comm.Get_size() # new: gives number of ranks in comm
            rank = mpi_comm.Get_rank()
            if size > 1:
                args.rank = rank
                args.world_size = size
                if os.getenv('MASTER_ADDR') is None:
                    os.environ['MASTER_ADDR'] = 'localhost'
                if os.getenv('MASTER_PORT') is None:
                    os.environ['MASTER_PORT'] = '12355'
            else:
                print(msg)
                args.distributed = False
                return
        except Exception as e:
            print(e)
            print("**mpi4py is not available, using mpirun will not run distributed mode")
            args.distributed = False
            return

    args.distributed = True
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    if args.hpu and args.world_size  > 1:
        args.dist_backend = 'hccl'
        os.environ["ID"] = str(args.rank % args.process_per_node )
        #not used currently
        os.environ["LOCAL_RANK"] = str(args.rank % args.process_per_node )
        import habana_frameworks.torch.core.hccl
        dist.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
    else:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                             world_size=args.world_size, rank=args.rank)

    setup_for_distributed(args.rank == 0)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        data, target = data.to(device), target.to(device)
        data = data.contiguous(memory_format=torch.channels_last)
        if args.use_lazy_mode:
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.use_lazy_mode:
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()

        optimizer.step()

        if args.use_lazy_mode:
            import habana_frameworks.torch.core as htcore

            htcore.mark_step()


def permute_params(model, to_filters_last, lazy_mode):
    import habana_frameworks.torch.core as htcore

    if htcore.is_enabled_weight_permute_pass() is True:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 4:
                if to_filters_last:
                    param.data = param.data.permute((2, 3, 1, 0))
                else:
                    param.data = param.data.permute(
                        (3, 2, 0, 1)
                    )  # permute RSCK to KCRS
    if lazy_mode:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()


def permute_momentum(optimizer, to_filters_last, lazy_mode):
    # Permute the momentum buffer before using for checkpoint
    import habana_frameworks.torch.core as htcore

    if htcore.is_enabled_weight_permute_pass() is True:
        return
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                if buf.ndim == 4:
                    if to_filters_last:
                        buf = buf.permute((2, 3, 1, 0))
                    else:
                        buf = buf.permute((3, 2, 0, 1))
                    param_state["momentum_buffer"] = buf

    if lazy_mode:
        import habana_frameworks.torch.core as htcore

        htcore.mark_step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch MNIST Example"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--hpu",
        action="store_true",
        default=False,
        help="Use hpu device",
    )
    parser.add_argument(
        "--use_lazy_mode",
        action="store_true",
        default=False,
        help="Enable lazy mode on hpu device, default eager mode",
    )
    parser.add_argument(
        "--hmp",
        dest="is_hmp",
        action="store_true",
        help="enable hmp mode",
    )
    parser.add_argument(
        "--hmp-bf16",
        default="./ops/ops_bf16_mnist.txt",
        help="path to bf16 ops list in hmp O1 mode",
    )
    parser.add_argument(
        "--hmp-fp32",
        default="./ops/ops_fp32_mnist.txt",
        help="path to fp32 ops list in hmp O1 mode",
    )
    parser.add_argument(
        "--hmp-opt-level",
        default="O1",
        help="choose optimization level for hmp",
    )
    parser.add_argument(
        "--hmp-verbose",
        action="store_true",
        help="enable verbose mode for hmp",
    )
    parser.add_argument(
        "--dl-worker-type",
        default="MP",
        type=lambda x: x.upper(),
        choices=["MT", "MP"],
        help="select multithreading or multiprocessing",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        metavar="N",
        help="number of total workers (default: 1)",
    )
    parser.add_argument(
        "--process-per-node",
        default=8,
        type=int,
        metavar="N",
        help="Number of process per node",
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="whether to enable distributed mode and run on multiple devices",
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        help="url used to set up distributed training",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.multiprocessing.set_start_method("spawn")
    if args.hpu:
        from habana_frameworks.torch.utils.library_loader import (
            load_habana_module,
        )

        load_habana_module()
        device = torch.device("hpu")
        # patch torch cuda functions that are being unconditionally invoked
        # in the multiprocessing data loader
        torch.cuda.current_device = lambda: None
        torch.cuda.set_device = lambda x: None

    if args.use_lazy_mode:
        os.environ["PT_HPU_LAZY_MODE"] = "1"
        import habana_frameworks.torch.core as htcore

    if args.is_hmp:
        from habana_frameworks.torch.hpex import hmp

        hmp.convert(
            opt_level=args.hmp_opt_level,
            bf16_file_path=args.hmp_bf16,
            fp32_file_path=args.hmp_fp32,
            isVerbose=args.hmp_verbose,
        )

    init_distributed_mode(args)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset1 = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset1
        )

        train_loader = torch.utils.data.DataLoader(
            dataset1,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            dataset1, **train_kwargs
        )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    if args.hpu:
        permute_params(model, True, args.use_lazy_mode)
        permute_momentum(optimizer, True, args.use_lazy_mode)

    if args.distributed and args.hpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            bucket_cap_mb=100,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

    if args.save_model:
        if args.hpu:
            torch.save(model.cpu().state_dict(), "mnist_cnn.pt")
        else:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
