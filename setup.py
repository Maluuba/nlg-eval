#!/usr/bin/env python

import logging
import os
import stat
import sys
import time
from zipfile import ZipFile

from setuptools import find_packages
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from pip._internal.req import parse_requirements
except:
    from pip.req import parse_requirements


def _download_file(d):
    import requests
    from tqdm import tqdm

    url, target_dir = d['url'], d['target_dir']
    filename = url[url.rfind('/') + 1:]
    target_path = os.path.join(target_dir, filename)
    if not os.path.exists(target_path):
        # Collect data 1MB at a time.
        chunk_size = 1 * 1024 * 1024
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        num_attempts = 3

        for attempt_num in range(num_attempts):
            try:
                print("Downloading {} to {}.".format(url, target_dir))
                r = requests.get(url, stream=True)
                r.raise_for_status()

                total = None
                length = r.headers.get('Content-length')
                if length is not None:
                    total = int(length) // chunk_size + 1

                with open(target_path, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                                      desc="{}".format(filename),
                                      total=total,
                                      unit_scale=True, mininterval=15, unit=" chunks"):
                        sys.stdout.flush()
                        f.write(chunk)
                break
            except:
                if attempt_num < num_attempts - 1:
                    wait_s = 1 * 60
                    logging.exception("Error downloading file, will retry in %ds.", wait_s)
                    # Wait and try to download later.
                    time.sleep(wait_s)
                else:
                    raise


def _post_setup():
    from multiprocessing import Pool
    from nltk.downloader import download
    download('punkt')

    data_path = os.getenv('NLGEVAL_DATA', 'nlgeval/data')

    path = 'nlgeval/word2vec/glove2word2vec.py'
    if os.path.exists(path):
        os.remove('nlgeval/word2vec/glove2word2vec.py')

    downloads = []

    if sys.version_info[0] == 2:
        downloads.append(dict(
            url='https://raw.githubusercontent.com/manasRK/glove-gensim/42ce46f00e83d3afa028fb6bf17ed3c90ca65fcc/glove2word2vec.py',
            target_dir='nlgeval/word2vec'
        ))
    else:
        downloads.append(dict(
            url='https://raw.githubusercontent.com/robmsmt/glove-gensim/dea5e55f449794567f12c79dc12b7f75339b18ba/glove2word2vec.py',
            target_dir='nlgeval/word2vec'
        ))

    setup_glove = not os.path.exists(os.path.join(data_path, 'glove.6B.300d.model.bin'))
    if setup_glove:
        downloads.append(dict(
            url='http://nlp.stanford.edu/data/glove.6B.zip',
            target_dir=data_path
        ))

    # Skip-thoughts data.
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/dictionary.txt',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/utable.npy',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/btable.npy',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz',
        target_dir=data_path
    ))
    downloads.append(dict(
        url='http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl',
        target_dir=data_path
    ))

    # multi-bleu.perl
    downloads.append(dict(
        url='https://raw.githubusercontent.com/moses-smt/mosesdecoder/b199e654df2a26ea58f234cbb642e89d9c1f269d/scripts/generic/multi-bleu.perl',
        target_dir='nlgeval/multibleu'
    ))

    # Limit the number of threads so that we don't download too much from the same source concurrently.
    pool = Pool(min(4, len(downloads)))
    pool.map(_download_file, downloads)
    pool.close()
    pool.join()

    if setup_glove:
        from nlgeval.word2vec.generate_w2v_files import generate
        z = ZipFile(os.path.join(data_path, 'glove.6B.zip'))
        z.extract('glove.6B.300d.txt', data_path)
        generate()
        for p in [
            os.path.join(data_path, 'glove.6B.zip'),
            os.path.join(data_path, 'glove.6B.300d.txt'),
            os.path.join(data_path, 'glove.6B.300d.model.txt'),
        ]:
            if os.path.exists(p):
                os.remove(p)

    path = 'nlgeval/multibleu/multi-bleu.perl'
    stats = os.stat(path)
    os.chmod(path, stats.st_mode | stat.S_IEXEC)


# Set up post install actions as per https://stackoverflow.com/a/36902139/1226799
class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        _post_setup()


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        _post_setup()


if __name__ == '__main__':
    requirements_path = 'requirements.txt'
    if sys.version_info[0] < 3:
        requirements_path = 'requirements_py2.txt'
    install_reqs = parse_requirements(requirements_path, session=False)
    reqs = [str(ir.req) for ir in install_reqs]

    setup(name='nlg-eval',
          version='2.0',
          description="Wrapper for multiple NLG evaluation methods and metrics.",
          author='Shikhar Sharma, Hannes Schulz, Justin Harris',
          author_email='shikhar.sharma@microsoft.com, hannes.schulz@microsoft.com, justin.harris@microsoft.com',
          url='https://github.com/Maluuba/nlg-eval',
          packages=find_packages(),
          include_package_data=True,
          scripts=['bin/nlg-eval'],
          install_requires=reqs,
          cmdclass={
              'develop': PostDevelopCommand,
              'install': PostInstallCommand,
          })
