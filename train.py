import gc
import os
import yaml
import random
import wandb
import torch
import logging
import argparse
import numpy as np
import torchattacks
import torch.nn as nn

# Custom
import losses
from dataloader import get_dataloader
from test import evaluation_robustness
from model import Purifier_Classifier, set_random_norm_mixed, set_conv
from utils import get_classifier, set_manual_seed, print_loss, colors, dict2namespace

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train(model, classifier, config):
    classifier.eval()
    
    ### Attack ###
    if config.TRAIN.ATTACK == "pgd":
        wb_attack = torchattacks.PGD(model, eps=config.TRAIN.EPS, alpha=2/255, steps=config.TRAIN.PGD_STEP)
        gb_attack = torchattacks.PGD(classifier, eps=config.TRAIN.EPS, alpha=2/255, steps=config.TRAIN.PGD_STEP)
    elif config.TRAIN.ATTACK == "eot-pgd":
        wb_attack = torchattacks.EOTPGD(model, eps=config.TRAIN.EPS, alpha=2/255, steps=config.TRAIN.PGD_STEP, 
                                              eot_iter=config.TRAIN.EOT_ITER, random_start=True)
        gb_attack = torchattacks.EOTPGD(classifier, eps=config.TRAIN.EPS, alpha=2/255, steps=config.TRAIN.PGD_STEP, 
                                              eot_iter=config.TRAIN.EOT_ITER, random_start=True)
    elif config.TRAIN.ATTACK == "fgsm":
        wb_attack = torchattacks.FGSM(model, eps=config.TRAIN.EPS)
        gb_attack = torchattacks.FGSM(classifier, eps=config.TRAIN.EPS)
    elif config.TRAIN.ATTACK == "cw":
        wb_attack = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
        gb_attack = torchattacks.CW(classifier, c=1, kappa=0, steps=100, lr=0.01)
    elif config.TRAIN.ATTACK == "aa":
        wb_attack = torchattacks.AutoAttack(model, norm='Linf', eps=config.TRAIN.EPS, version='standard',
                                                  n_classes=10, seed=config.TRAIN.SEED, verbose=False)
        gb_attack = torchattacks.AutoAttack(classifier, norm='Linf', eps=config.TRAIN.EPS, version='standard',
                                                    n_classes=10, seed=config.TRAIN.SEED, verbose=False)
    else:
        raise Exception("Attack method is wrong. It should be pgd or aa.")
    
    ### wandb setting ###
    os.environ['WANDB_API_KEY'] = "0ee23525f6f4ddbbab74086ddc0b2294c7793e80"
    wandb.init(project=config.PROJECT, entity="psj", name=config.EXP, tags=[config.DATASET])
    wandb.config.update(config)
    
    ### Optimizers ###
    optimizer = torch.optim.Adam(model.purifier.parameters(), lr=config.TRAIN.LR, 
                                 betas=(config.TRAIN.B1, config.TRAIN.B2), weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    ### Load checkpoint ###
    if config.TRAIN.RESUME_EPOCH is not None:
        if config.TRAIN.RESUME_EPOCH == -1:
            resume_name = "final"
        else:
            resume_name = f"epoch{config.TRAIN.RESUME_EPOCH}"
        norm_names = ""
        for norm in config.NORM_LIST:
            norm_names += f"{norm}_" 
            
        conv_names = ""
        for conv in config.CONV_LIST:
            conv_names += f"{conv}_"
        ckpt_path = os.path.join(config.RESULT_PATH, 
                                 f"{config.DATASET}_{norm_names}{conv_names}{resume_name}.pth")
        checkpoint = torch.load(ckpt_path)
        model.purifier.load_state_dict(checkpoint["purifier"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = checkpoint["lr"]
            current_lr = g["lr"]
        logging.info("[*] Load the model at step %d" % checkpoint["iteration"])
        start_iteration = checkpoint["iteration"]
    else:
        start_iteration = 0
        
    ### Dataloader ###
    train_loader = get_dataloader(dataset_name=config.DATASET, which="train", 
                                  subsample=config.TRAIN_SUBSAMPLE, 
                                  batch_size=config.TRAIN.BATCH_SIZE, config=config)
    test_loader  = get_dataloader(dataset_name=config.DATASET, which="val", 
                                  subsample=config.TRAIN.TEST_SUBSAMPLE,
                                  batch_size=config.TRAIN.EVAL_BATCH_SIZE, config=config)
    
    mse_loss = nn.MSELoss(size_average=False).to(device)
    losses_dict = {}
    train_loader_iterator = iter(train_loader)
    total_iteration = config.TRAIN.EPOCHS * len(train_loader)
    
    current_lr = config.TRAIN.LR
    
    ### Training start ###
    for idx, iteration in enumerate(range(start_iteration, total_iteration)):
        try:
            (x, y) = next(train_loader_iterator)
        except StopIteration:
            np.random.seed()  # Ensure randomness
            # Some cleanup
            train_loader_iterator = None
            torch.cuda.empty_cache()
            gc.collect()
            train_loader_iterator = iter(train_loader)
            (x, y) = next(train_loader_iterator)
        
        optimizer.zero_grad()
        
        x, y = x.to(device), y.to(device)
        model.train()
        
        set_random_norm_mixed(model, config.BLOCK_WISE)
        purifier_adv = wb_attack(x, y)
        classifier_adv = gb_attack(x, y)
        
        total_loss = 0
        set_random_norm_mixed(model, config.BLOCK_WISE)
        pred_cln, x_cln = model(x,            True)
        pred_pur, x_pur = model(purifier_adv, True)
        losses_dict["L2"] = mse_loss(x, x_pur) / (x.size(0)*2)
        recon_loss = 0
        for name in config.CONV_LIST:
            set_conv(model, name)
            _, recon = model(purifier_adv, True)
            recon_loss += mse_loss(x, recon) / (x.size(0)*2)
        recon_loss /= len(config.CONV_LIST)
        total_loss += losses_dict["L2"] + recon_loss
        total_loss.backward()
        optimizer.step()
        
        wandb.log(losses_dict.copy())
        
        ### Calculate train accuracy ###
        with torch.no_grad():
            for _ in range(config.NUM_ITER-1):
                set_random_norm_mixed(model, config.BLOCK_WISE)
                pred, *_ = model(x, True)
                pred_cln += pred
                pred, x_pur = model(purifier_adv, True)
                pred_pur += pred
            
            prediction_dict = {
                "nat": classifier(x), 
                "cln": pred_cln / config.NUM_ITER, 
                "adv": classifier(classifier_adv), 
                "rob": pred_pur / config.NUM_ITER,
                }
            correct_dict = {}
            log_dict = {}
            for name in prediction_dict.keys():
                correct_dict[name] = 0
                
            for k, v in prediction_dict.items():
                _, y_pred = torch.max(v.data, 1)
                correct = (y_pred == y).sum().item()
                log_dict[f"train_{k}_acc"] = (correct/x.size(0)) * 100
        
        wandb.log({"iteration": iteration})
        
        ### Print the training process ###
        if iteration % config.TRAIN.PRINT_FREQ == 0:
            logging.info("")
            logging.info("[Epoch %d/%d] [Iteration %d/%d]" % 
                         (iteration // (len(train_loader)),
                          config.TRAIN.EPOCHS, 
                          iteration, 
                          total_iteration))
            print_loss({**losses_dict, **log_dict})
            wandb.log(log_dict)

        ### Learning rate decay
        if iteration % len(train_loader) == 0 and (iteration // (len(train_loader)) in config.TRAIN.LR_DECAY_EPOCH):
            for g in optimizer.param_groups:
                g["lr"] *= config.TRAIN.LR_DECAY_RATIO
                current_lr = g["lr"]
        wandb.log({"lr": current_lr})
        
        ### Save the model per each epoch ###
        if (iteration % len(train_loader) == 0) and \
            ((iteration // len(train_loader)) % config.TRAIN.SAVE_FREQ == 0):
            norm_names = ""
            for norm in config.NORM_LIST:
                norm_names += f"{norm}_" 
            conv_names = ""
            for conv in config.CONV_LIST:
                conv_names += f"{conv}_"
            ckpt_name = f"{config.DATASET}_{norm_names}{conv_names}epoch{iteration // len(train_loader)}.pth"
            ckpt_path = os.path.join(config.RESULT_PATH, ckpt_name)
            torch.save({"purifier"  :model.purifier.state_dict(),
                        "optimizer" :optimizer.state_dict(),
                        "iteration" :iteration,
                        "lr"        :current_lr,
                        },
                       ckpt_path)
        
        ### Evaluation every specified iteration ###
        if iteration % config.TRAIN.EVAL_FREQ == 0:
            model.eval()
            atk_acc_dict, images = evaluation_robustness(model, classifier, test_loader, [config.TRAIN.ATTACK], config)
            for atk_name, acc_dict in atk_acc_dict.items():
                logging.info("")
                logging.info(colors.YELLOW + f"[Evaluation on test subset of {config.DATASET}]" + colors.WHITE)
                for name, acc in acc_dict.items():
                    logging.info(f"{name:>18s} | {acc:05.2f}%")
                wandb.log(acc_dict)
            
            torch.cuda.empty_cache()
            
            ### Save image ###
            for name, img in images.items():
                wandb.log({name: [wandb.Image(img[:16])]})

        torch.cuda.empty_cache()
    
    logging.info("[*] Saving the final model...")
    norm_names = ""
    for norm in config.NORM_LIST:
        norm_names += f"{norm}_" 
        
    conv_names = ""
    for conv in config.CONV_LIST:
        conv_names += f"{conv}_"
    ckpt_path = os.path.join(config.RESULT_PATH, f"{config.DATASET}_{norm_names}{conv_names}final.pth")
    torch.save({"purifier"  : model.purifier.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "iteration" : iteration,
                "lr"        : current_lr,
                },
               ckpt_path)
    logging.info("[*] Training finished.")
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()

    # Training parameter
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        dict_ = yaml.safe_load(f)
    config = dict2namespace(dict_)
    
    ### Basic configuration ###
    os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % config.GPU
    device = 'cuda:%d' % config.GPU if torch.cuda.is_available() else 'cpu'
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    
    ### Set manual seed ###
    set_manual_seed(config.TRAIN.SEED)
    
    ### Result path ###
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    config.RESULT_PATH = os.path.join(config.RESULT_PATH, config.PROJECT, config.EXP)
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    
    model = Purifier_Classifier(config).to(device)
    classifier = get_classifier(config).to(device)
    
    """ Training """
    train(model, classifier, config)
    
    """ Test """
    logging.info("[!] Evaluation ...")
    test_loader = get_dataloader(dataset_name= config.DATASET, which="val", 
                                    subsample   = config.TEST.TEST_SUBSAMPLE, 
                                    batch_size  = config.TEST.BATCH_SIZE,
                                    config      = config,)
    atk_acc_dict, _ = evaluation_robustness(model, classifier, test_loader, config.TEST.ATTACKS, config)
    
    for atk_name, acc_dict in atk_acc_dict.items():
        logging.info("")
        logging.info(colors.YELLOW + f"[Evaluation results on testset of {config.DATASET}]" + colors.WHITE)
        print_accuracies = {"Attack":              atk_name,
                            "Classifier":          config.CLASSIFIER,
                            }
        print_accuracies = {**print_accuracies, **acc_dict}
        
        for name, acc in print_accuracies.items():
            if type(acc) == float:
                logging.info(f"{name:>20s} | {acc:05.2f}%")
            else:
                logging.info(f"{name:>20s} | {acc}")
    
if __name__ == "__main__":
    main()