"""Main file for running denoising-diffusion-flax.
"""

#import scipy.optimize

import flax

import train, sampling

import jax
from absl import app
from absl import flags
from absl import logging
from clu import platform
from ml_collections import config_flags



import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training or sampling hyperparameter configuration.',
    lock_config=True)

flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.DEFINE_string("wandb_artifact", None, "the wandb artifact reference for logged model")
flags.DEFINE_string("mode", "train", "Running mode: train or sample")

def main(argv):
  if len(argv) > 3:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())


  if FLAGS.mode == "train":
      train.train_seg(FLAGS.config, FLAGS.workdir, FLAGS.wandb_artifact)
  if FLAGS.mode == "sample":
      train.sample_seg(FLAGS.config, FLAGS.workdir, FLAGS.wandb_artifact)
  if FLAGS.mode == "sample_ddim":
      train.sample_seg_ddim(FLAGS.config, FLAGS.workdir, FLAGS.wandb_artifact)
  else:
      raise ValueError(f"Mode {FLAGS.mode} not recognized.")

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)