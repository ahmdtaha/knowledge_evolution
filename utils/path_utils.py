import os
import getpass
import pathlib
import os.path as osp
from datetime import datetime

username = getpass.getuser()

def get_checkpoint_dir():

    project_name = osp.basename(osp.abspath('./'))
    if username == 'ahmdtaha':
        ckpt_dir = '/vulcanscratch/ahmdtaha/checkpoints'
    elif username == 'ataha':
        ckpt_dir = '/mnt/data/checkpoints'
    elif username == 'ahmedtaha':
        ckpt_dir = '/Users/ahmedtaha/Documents/checkpoints'
    else:
        raise NotImplementedError('Invalid username {}'.format(username))

    assert osp.exists(ckpt_dir),('{} does not exists'.format(ckpt_dir))

    ckpt_dir = f'{ckpt_dir}/{project_name}'
    return ckpt_dir

def get_pretrained_ckpt(model_name):

    if username == 'ahmdtaha':
        pretrained_dir = '/vulcanscratch/ahmdtaha/pretrained'
    elif username == 'ataha':
        pretrained_dir = '/mnt/data/pretrained'
    elif username == 'ahmedtaha':
        pretrained_dir = '/Users/ahmedtaha/Documents/pretrained'
    else:
        raise NotImplementedError('Invalid username {}'.format(username))

    assert osp.exists(pretrained_dir),('{} does not exists'.format(pretrained_dir))

    if model_name == 'vgg_tensorpack':
        ckpt_path = 'tensorpack/vgg16.npz'
    elif model_name == 'resnet_v1_50':
        ckpt_path = 'resnet_v1_50/resnet_v1_50.ckpt'
    elif model_name == 'resnet_v1_101':
        ckpt_path = 'resnet_v1_101/resnet_v1_101.ckpt'
    elif model_name == 'inception_v1':
        ckpt_path = 'inception_v1/inception_v1.ckpt'
    elif model_name == 'densenet169':
        ckpt_path = 'tf-densenet169/tf-densenet169.ckpt'
    else:
        raise NotImplementedError('Invalid pretrained model name {}'.format(model_name))

    pretrained_ckpt = '{}/{}'.format(pretrained_dir,ckpt_path)
    return pretrained_ckpt

def get_datasets_dir(dataset_name):

    if username == 'ahmdtaha':
        datasets_dir = '/scratch0/ahmdtaha/datasets'
    elif username == 'ataha':
        datasets_dir = '/mnt/data/datasets'
    elif username == 'ahmedtaha':
        datasets_dir = '/Users/ahmedtaha/Documents/datasets'
    else:
        raise NotImplementedError('Invalid username {}'.format(username))

    assert osp.exists(datasets_dir),('{} does not exists'.format(datasets_dir))
    # print(dataset_name)
    if dataset_name == 'CUB200' or dataset_name == 'CUB200_RET':
        dataset_dir = 'CUB_200_2011'
    elif dataset_name == 'CARS_RET':
        dataset_dir = 'stanford_cars'
    elif dataset_name == 'stanford':
        dataset_dir = 'Stanford_Online_Products'
    elif dataset_name == 'imagenet':
        dataset_dir = 'imagenet/ILSVRC/Data/CLS-LOC'
    elif dataset_name == 'market':
        dataset_dir = 'Market-1501-v15.09.15'
    elif dataset_name == 'Flower102' or dataset_name == 'Flower102Pytorch':
        dataset_dir = 'flower102'
    elif dataset_name == 'HAM':
        dataset_dir = 'HAM'
    elif dataset_name == 'FCAM':
        dataset_dir = 'FCAM'
    elif dataset_name == 'FCAMD':
        dataset_dir = 'FCAMD'
    elif dataset_name == 'Dog120':
        dataset_dir = 'stanford_dogs'
    elif dataset_name in ['MIT67','MINI_MIT67']:
        dataset_dir = 'mit67'
    elif dataset_name == 'Aircraft100':
        dataset_dir = 'aircrafts'
    elif dataset_name == 'ImageNet':
        if username == 'ataha':
            dataset_dir = 'imagenet/ILSVRC/Data/CLS-LOC'
        else:
            dataset_dir = 'imagenet'
    else:
        raise NotImplementedError('Invalid dataset name {}'.format(dataset_name))

    datasets_dir = '{}/{}'.format(datasets_dir, dataset_dir)

    return datasets_dir



def get_directories(args,generation):
    # if args.config_file is None or args.name is None:
    if args.config_file is None and args.name is None:
        raise ValueError("Must have name and config")

    # config = pathlib.Path(args.config_file).stem
    config = args.name
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"{get_checkpoint_dir()}/{args.name}/gen_{generation}/split_rate={args.split_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{args.name}/gen_{generation}/split_rate={args.split_rate}"
        )
    
    def _run_dir_exists(run_base_dir):
        log_base_dir = run_base_dir / "logs"
        ckpt_base_dir = run_base_dir / "checkpoints"

        return log_base_dir.exists() or ckpt_base_dir.exists()

   # if _run_dir_exists(run_base_dir):
    rep_count = 0
    while _run_dir_exists(run_base_dir / '{:04d}_g{:01d}'.format(rep_count,args.gpu)):
        rep_count += 1

    # date_time_int = int(datetime.now().strftime('%Y%m%d%H%M'))
    run_base_dir = run_base_dir / '{:04d}_g{:01d}'.format(rep_count,args.gpu)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


if __name__ == '__main__':
    print(get_checkpoint_dir('test_exp'))
    print(get_pretrained_ckpt('vgg_tensorpack'))
    print(get_datasets_dir('cub'))