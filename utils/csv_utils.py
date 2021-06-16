import os
import time
import pathlib
from utils import path_utils

def write_cls_result_to_csv(**kwargs):
    name = kwargs.get('name')
    if '/' in name:
        exp_name = name.split('/')[0]
        results = pathlib.Path(os.path.join(path_utils.get_checkpoint_dir(),exp_name, "{}.csv".format(exp_name)))
    else:
        results = pathlib.Path(os.path.join(path_utils.get_checkpoint_dir(), "{}.csv".format(name) ))

    if not results.exists():
        results.write_text(
            "Date Finished, "
            # "Base Config, "
            "Name, "
            "Split Rate, "
            "Bias Split Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            
            "Current Tst Top 1, "
            "Current Tst Top 5, "
            "Best Tst Top 1, "
            "Best Tst Top 5, "
            
            "Best Trn Top 1, "
            "Best Trn Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                # "{base_config}, "
                "{name}, "
                "{split_rate}, "
                "{bias_split_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                
                "{last_tst_acc1:.02f}, "
                "{last_tst_acc5:.02f}, "
                "{best_tst_acc1:.02f}, "
                "{best_tst_acc5:.02f}, "
                
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


def write_ret_result_to_csv(**kwargs):
    name = kwargs.get('name')
    name_prefix = kwargs.get('name_prefix')

    if '/' in name:
        exp_name = name.split('/')[0]
        if name_prefix is None:
            results = pathlib.Path(os.path.join(path_utils.get_checkpoint_dir(), exp_name, "{}.csv".format(exp_name)))
        else:
            results = pathlib.Path(os.path.join(path_utils.get_checkpoint_dir(), exp_name, "{}_{}.csv".format(name_prefix,exp_name)))
    else:
        results = pathlib.Path(os.path.join(path_utils.get_checkpoint_dir(), "{}.csv".format(name)))

    if not results.exists():
        results.write_text(
            "Date Finished, "
            # "Base Config, "
            "Name, "
            "Split Rate, "
            "Bias Split Rate, "
            "NMI,"
            "R@1,"
            "R@2,"
            "R@4,"
            "R@8,"
            "R@16,"
            "R@32\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                # "{base_config}, "
                "{name}, "
                "{split_rate}, "
                "{bias_split_rate}, "
                "{NMI:.03f}, "
                "{R_1:.02f}, "
                "{R_2:.02f}, "
                "{R_4:.02f}, "
                "{R_8:.02f}, "
                "{R_16:.02f}, "
                "{R_32:.02f}\n"
            ).format(now=now, **kwargs)
        )
