#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
from __future__ import division

import atexit
import logging
import os
import subprocess
import sys
import threading

import psutil

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:

    def __init__(self):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                v = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(dec(v.strip())))
                except:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))
                    # You can try uncommenting the next code line to show stderr from the Meteor JAR.
                    # If the Meteor JAR is not writing to stderr, then the line will just hang.
                    # sys.stderr.write("Error from Meteor:\n{}".format(self.meteor_p.stderr.read()))
                    raise
            score = float(dec(self.meteor_p.stdout.readline()).strip())

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats 
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        self.close()
