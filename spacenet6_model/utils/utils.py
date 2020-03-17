import git
import json


def config_filename():
    """
    """
    return 'config.yml'


def experiment_subdir(exp_id):
    """
    """
    assert 0 <= exp_id <= 9999
    return f'exp_{exp_id:04d}'


def ensemble_subdir(exp_ids):
    """
    """
    exp_ids_ = exp_ids.copy()
    exp_ids_.sort()
    subdir = 'exp'
    for exp_id in exp_ids_:
        subdir += f'_{exp_id:04d}'
    return subdir


def git_filename():
    """
    """
    return 'git.json'


def weight_best_filename():
    """
    """
    return 'model_best.pth'


def weight_epoch_filename(epoch):
    """
    """
    assert 0 <= epoch <= 9999
    return f'model_{epoch:04d}.pth'


def dump_git_info(path):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    git_info = {
        'version': '0.0.0',
        'sha': sha
    }

    with open(path, 'w') as f:
        json.dump(
            git_info,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=False,
            separators=(',', ': ')
        )


def get_roi_mask(sar_image):
    mask = sar_image.sum(axis=-1) > 0.0
    return mask


def crop_center(array, crop_wh):
    """
    """
    _, h, w = array.shape
    crop_w, crop_h = crop_wh
    assert w >= crop_w
    assert h >= crop_h

    left = (w - crop_w) // 2
    right = crop_w + left
    top = (h - crop_h) // 2
    bottom = crop_h + top

    return array[:, top:bottom, left:right]
