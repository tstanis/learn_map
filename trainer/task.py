import argparse
import glob
import os

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

from tensorflow.python.lib.io import file_io

import trainer.learn as learn

INPUT_SIZE = 55
CLASS_SIZE = 2

# CHUNK_SIZE specifies the number of lines
# to read in case the file is very large
CHUNK_SIZE = 5000
CHECKPOINT_FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
LEARN_MAP_MODEL = 'learn_map.h5'

def train_and_evaluate(args):
    print(str(args))
    # Unhappy hack to workaround h5py not being able to write to GCS.
    # Force snapshots and saves to local filesystem, then copy them over to GCS.
    checkpoint_path = CHECKPOINT_FILE_PATH
    if not args.job_dir.startswith('gs://'):
        checkpoint_path = os.path.join(args.job_dir, checkpoint_path)

    # Model checkpoint callback.
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        period=args.checkpoint_epochs,
        mode='min')

    
    log_dir=os.path.join(args.job_dir, 'logs')

    # Tensorboard logs callback.
    tb_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
  #      update_freq=1000,
        embeddings_freq=0)

    callbacks = []
    if args.tensorboard:
      callbacks.append(tb_log)

    model = None
    if args.load_model:
      print("Loading Module ...")
      model_file = file_io.FileIO(args.load_model, mode='rb')
      temp_model_location = LEARN_MAP_MODEL
      temp_model_file = open(temp_model_location, 'wb')
      temp_model_file.write(model_file.read())
      temp_model_file.close()
      model_file.close()
      model = load_model(LEARN_MAP_MODEL)

    #print("Train: " + str(args.train))
    if args.train:
      model = learn.train(model, args.grid_size, callbacks, log_dir, args.train_steps, args.train_batch_size, args.num_epochs, args.eval_batch_size)

    if args.evaluate:
      learn.evaluate(model, args.grid_size, log_dir, args.eval_batch_size)

    if args.navigate:
      learn.draw_navigation(model, args.grid_size)

    if args.train:
      print("Saving: " + LEARN_MAP_MODEL)
      model.save(LEARN_MAP_MODEL)
      with file_io.FileIO(LEARN_MAP_MODEL, mode='rb') as input_f:
          with file_io.FileIO(args.job_dir + '/' + LEARN_MAP_MODEL, mode='wb+') as output_f:
              output_f.write(input_f.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      type=str,
      help='GCS or local dir to write checkpoints and export model',
      default='/tmp/learn_map')
    parser.add_argument(
      '--train-steps',
      type=int,
      default=100,
      help="""\
        Maximum number of training steps to perform
        Training steps are in the units of training-batch-size.
        So if train-steps is 500 and train-batch-size if 100 then
        at most 500 * 100 training instances will be used to train.""")
    parser.add_argument(
      '--train-batch-size',
      type=int,
      default=40,
      help='Batch size for training steps')
    parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=40,
      help='Batch size for evaluation steps')
    parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.003,
      help='Learning rate for SGD')
    parser.add_argument(
      '--eval-num-epochs',
      type=int,
      default=1,
      help='Number of epochs during evaluation')
    parser.add_argument(
      '--num-epochs',
      type=int,
      default=20,
      help='Maximum number of epochs on which to train')
    parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=5,
      help='Checkpoint per n training epochs')
    parser.add_argument(
      '--grid-size',
      type=int,
      default=10,
      help='Size of the board to run')
    parser.add_argument(
      '--tensorboard',
      type=bool,
      default=False,
      help='Whether to log to tensorboard')
    parser.add_argument(
      '--train',
      type=bool,
      default=False,
      help='Whether to train or load existing model')
    parser.add_argument(
      '--evaluate',
      type=bool,
      default=False,
      help='Whether to evaluate model')
    parser.add_argument(
      '--navigate',
      type=bool,
      default=False,
      help='Whether to navigate')
    parser.add_argument(
      '--load-model',
      type=str,
      default=None,
      help="Model to start with")
    args, _ = parser.parse_known_args()
    train_and_evaluate(args)