import time
import os
import json
from models.model_loader import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep_notes import DataPreparation
import utils.arg_parser
from utils.logger import Logger
from utils.misc import get_split_str

import torch

if __name__ == '__main__':

    # Original comment: Parse arguments
    # the default is using the COCO data and the LRCN model. For using CUB and GVE, set the arguments in the command
    # line
    args = utils.arg_parser.get_args()

    # Original comment: Print arguments
    utils.arg_parser.print_args(args)

    # cuda is used to run the training / evaluation on the gpu. If it is not available,
    # the cpu is used instead (the same effect is caused by args.disable_cuda)
    device = torch.device('cuda:{}'.format(args.cuda_device) if
                          torch.cuda.is_available() and not args.disable_cuda else 'cpu')

    job_string = time.strftime("{}-{}-D%Y-%m-%d-T%H-%M-%S-G{}".format(args.model, args.dataset, args.cuda_device))

    job_path = os.path.join(args.checkpoint_path, job_string)

    # Original comment: Create new checkpoint directory
    os.makedirs(job_path)

    # Save job arguments
    with open(os.path.join(job_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    print("Preparing Data ...")
    # in the following, the dataset is loaded (CUB or COCO). split will be 'train' if args.train == True;
    # else it will be either 'test' or 'val', depending on args.eval_ckpt and args.dataset (see comments in
    # utils/misc.py)
    split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
    # for details about the dataset loading process: see data_prep.py, coco_dataset.py, cub_dataset.py
    data_prep = DataPreparation(args.dataset, args.data_path)
    dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
                                                            batch_size=args.batch_size, num_workers=args.num_workers)
    if args.train:
        # get the validation dataset, which will later be used for evaluation after each training epoch
        val_dataset, val_data_loader = data_prep.get_dataset_and_loader('val',
                                                                        args.pretrained_model,
                                                                        batch_size=args.batch_size,
                                                                        num_workers=args.num_workers)

    print()

    print("Loading Model ...")
    ml = ModelLoader(args, dataset)
    model = getattr(ml, args.model)()
    print(model, '\n')

    # if we are not training, get the weights and training states from the checkpoint
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

    trainer_creator = getattr(TrainerLoader, args.model)
    trainer = trainer_creator(args, model, dataset, data_loader, logger, device)
    if args.train:
        # evaluator is for validation dataset. We will use it after each epoch
        evaluator = trainer_creator(args, model, val_dataset, val_data_loader,
                                    logger, device)
        # the evaluator is not used for training
        evaluator.train = False

    if args.train:
        print("Training ...")
    else:
        print("Evaluating ...")
        vars(args)['num_epochs'] = 1

    # OWN CODE
    # suppress evaluation (to avoid errors)
    if args.train and args.without_eval:
        evaluator.REQ_EVAL = False

    # the following loop goes through the specified number of epochs, either training and evaluating or only
    # evaluating
    max_score = 0
    while trainer.curr_epoch < args.num_epochs:
        if args.train:
            # here the actual training happens; for details see train.trainer_loader and train.lrcn_trainer
            trainer.train_epoch()

            # Original comment: Eval & Checkpoint
            checkpoint_name = "ckpt-e{}".format(trainer.curr_epoch)
            checkpoint_path = os.path.join(job_path, checkpoint_name)

            # the model is set to evaluation mode with .eval(), then it is evaluated, and then set back to
            # training settings with .train()
            model.eval()
            result = evaluator.train_epoch()

            # MODIFIED CODE
            # the original code had score = result if REQ_EVAL is False, which will lead to an error because the
            # following code (logging and comparing to max_score) expects a number
            # hence, we only calculate a score if REQ_EVAL is True, otherwise we only store the state_dict
            if evaluator.REQ_EVAL:
                score = val_dataset.eval(result, checkpoint_path)
                model.train()

                # write to log
                logger.scalar_summary('score', score, trainer.curr_epoch)

                # Original comment: Save the models
                checkpoint = {'epoch': trainer.curr_epoch,
                              'max_score': max_score,
                              'optimizer': trainer.optimizer.state_dict()}
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
                model.train()

                # Original comment: Save the models
                checkpoint = {'epoch': trainer.curr_epoch,
                              'optimizer': trainer.optimizer.state_dict()}
                checkpoint_path += ".pth"
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(checkpoint, os.path.join(job_path,
                                                    "training_checkpoint.pth"))

        else:
            # here, only the evaluation is done.
            # "Evaluation" means general evaluation of the model here (i.e., BLEU,
            # METEOR, CIDEr scores), not the evaluation of issues and captions, which is
            # done separately

            # when train is false, train_epoch() returns a list of generated sentences to be evaluated.
            result = trainer.train_epoch()
            if trainer.REQ_EVAL:
                # evaluation of results
                score = dataset.eval(result, "results")
