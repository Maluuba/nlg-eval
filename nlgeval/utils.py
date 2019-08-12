import click
import json
import os

from xdg import XDG_CONFIG_HOME


class InvalidDataDirException(Exception):
  pass


def get_data_dir():
    if os.environ.get('NLGEVAL_DATA'):
        if not os.path.exists(os.environ.get('NLGEVAL_DATA')):
            click.secho("NLGEVAL_DATA variable is set but points to non-existent path.", fg='red', err=True)
            raise InvalidDataDirException()
        return os.environ.get('NLGEVAL_DATA')
    else:
        try:
            cfg_file = os.path.join(XDG_CONFIG_HOME, 'nlgeval', 'rc.json')
            with open(cfg_file, 'rt') as f:
                rc = json.load(f)
                if not os.path.exists(rc['data_path']):
                    click.secho("Data path found in {} does not exist: {} " % (cfg_file, rc['data_path']), fg='red', err=True)
                    click.secho("Run `nlg-eval --setup DATA_DIR' to download or set $NLGEVAL_DATA to an existing location",
                                fg='red', err=True)
                    raise InvalidDataDirException()
                return rc['data_path']
        except:
            click.secho("Could not determine location of data.", fg='red', err=True)
            click.secho("Run `nlg-eval --setup DATA_DIR' to download or set $NLGEVAL_DATA to an existing location", fg='red',
                        err=True)
            raise InvalidDataDirException()
