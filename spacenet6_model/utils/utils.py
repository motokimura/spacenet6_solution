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
