import time
import os
import json

import sys 

from models.model_loader_reduced import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep_notes import DataPreparation
import utils.arg_parser
from utils.logger import Logger
from utils.misc import get_split_str

import torch

if __name__ == '__main__':

    # Parse arguments
    args = utils.arg_parser.get_args()

    # NOTE: The default is using COCO. For using CUB, change arguments here:
    # args.model = 'gve'
    # args.dataset = 'cub'

    # Print arguments
    utils.arg_parser.print_args(args)

    # NOTE: cuda is used to run the training / evaluation on the gpu. If it is not available,
    # the cpu is used instead (the same effect is caused by args.disable_cuda)

    device = torch.device('cuda:{}'.format(args.cuda_device) if
            torch.cuda.is_available() and not args.disable_cuda else 'cpu')

    job_string = time.strftime("{}-{}-D%Y-%m-%d-T%H-%M-%S-G{}".format(args.model, args.dataset, args.cuda_device))

    job_path = os.path.join(args.checkpoint_path, job_string)


    # Create new checkpoint directory
    #if not os.path.exists(job_path):
    os.makedirs(job_path)

    # Save job arguments
    with open(os.path.join(job_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # NOTE: In the following, the dataset is loaded (CUB or COCO). split will be 'train' if args.train == True;
    # else it will be either 'test' or 'val', depending on args.eval_ckpt and args.dataset
    # NOTE AND TODO: What role exactly does args.eval_ckpt play here?
    # If args.train == True, additionally, there will be used a validation dataset for evaluation
    # Details about the dataset loading process: see data_prep.py, coco_dataset.py, cub_dataset.py

    # Data preparation
    print("Preparing Data ...")
    split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
    data_prep = DataPreparation(args.dataset, args.data_path)
    dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
            batch_size=args.batch_size, num_workers=args.num_workers)
    if args.train:
        val_dataset, val_data_loader = data_prep.get_dataset_and_loader('val',
                args.pretrained_model, batch_size=args.batch_size, num_workers=args.num_workers)

    # TODO: If eval + checkpoint load validation set

    print()

    # NOTE AND TODO: No comments on the following two sections yet

    print("Loading Model ...")
    ml = ModelLoader(args, dataset)
    model = getattr(ml, args.model)()
    print(model, '\n')

    # If we are not training, get the weights and training states from the checkpoint
    if not args.train:
        print("Loading Model Weights ...")
        evaluation_state_dict = torch.load(args.eval_ckpt, map_location=device)
        model_dict = model.state_dict(full_dict=True)
        model_dict.update(evaluation_state_dict)
        model.load_state_dict(model_dict)
        model.eval()

    if args.train:
        val_dataset.set_label_usage(dataset.return_labels)

    # Create logger. tensorboard --logdir=path/to/logs allows us to visualize batch and epoch loss
    logger = Logger(os.path.join(job_path, 'logs'))

    # NOTE AND TODO: No comment on the following section yet

    # Get trainer
    trainer_creator = getattr(TrainerLoader, args.model)
    trainer = trainer_creator(args, model, dataset, data_loader, logger, device)
    if args.train:
        evaluator = trainer_creator(args, model, val_dataset, val_data_loader,
            logger, device)
        evaluator.train = False

    if args.train:
        print("Training ...")
    else:
        print("Evaluating ...")
        vars(args)['num_epochs'] = 1

    # NOTE: The following loop goes throught the specified number of epochs, either training and evaluating or only
    # evaluating

    # Start training/evaluation
    max_score = 0
    while trainer.curr_epoch < args.num_epochs:
        if args.train:
            # NOTE: Here the actual training happens; for details see train.trainer_loader and train.lrcn_trainer
            trainer.train_epoch()

            # Eval & Checkpoint
            checkpoint_name = "ckpt-e{}".format(trainer.curr_epoch)
            checkpoint_path = os.path.join(job_path, checkpoint_name)

            # NOTE: the model is set to evaluation mode with .eval(), then it is evaluated, and then set back to
            # training settings with .train()
            model.eval()
            result = evaluator.train_epoch()
            if evaluator.REQ_EVAL:
                score = val_dataset.eval(result, checkpoint_path)
            else:
                score = result
            model.train()

            # NOTE: write to log
            logger.scalar_summary('score', score, trainer.curr_epoch)

            # NOTE AND TODO: The checkpoints do not work for me. There are two possibilities:
            # 1. either it may be an issue of the Python version,
            # 2. or they are indeed not correctly implemented in the code (in which case
            # we would probably not be able to fix it, and could just comment them out)
            # Could you try whether this part works for you (with Python 3.7)?

            # TODO: Eval model
            # Save the models
            checkpoint = {'epoch': trainer.curr_epoch,
                          'max_score': max_score,
                          'optimizer' : trainer.optimizer.state_dict()}
            checkpoint_path += ".pth"
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(checkpoint, os.path.join(job_path,
                "training_checkpoint.pth"))
            # Check whether this is the best score we have obtained so far. If so, save as such
            if score > max_score:
                max_score = score
                link_name = "best-ckpt.pth"
                link_path = os.path.join(job_path, link_name)
                if os.path.islink(link_path):
                    os.unlink(link_path)
                dir_fd = os.open(os.path.dirname(link_path), os.O_RDONLY)
                os.symlink(os.path.basename(checkpoint_path), link_name, dir_fd=dir_fd)
                os.close(dir_fd)

        else:

            # NOTE: here, only the evaluation is done, training is not necessary
            # "Evaluation" means general evaluation of the model here (i.e., BLEU,
            # METEOR, CIDEr scores), not the evaluation of issues and captions, which is
            # done separately
            # NOTE AND TODO: Why do we have a "trainer" for evaluation, and not an "evaluator"?
            # (while for args.train we have a trainer for training and an evaluator for evaluation)
            # it seems that for args.train, the evaluation is done on the validation dataset,
            # while for args.train == False, it is done on "dataset" (the training data?)

            result = trainer.train_epoch()
            if trainer.REQ_EVAL:
                score = dataset.eval(result, "results")

    # NOTE AND TODO: What does 'sc' mean here? Is it another kind of model? Why is it relevant here? See answer below

    # NOTE: If what we are testing/validating is the SentenceClassifier, save results in a JSON file #TODO I am unsure
    if not args.train and args.model == 'sc':
        with open('results.json', 'w') as f:
            json.dump(result, f)

