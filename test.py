import os
import yaml
import time
import torch
import logging
import argparse
import torchattacks

from tqdm import tqdm

from dataloader import get_dataloader
from utils import get_classifier, set_manual_seed, colors, dict2namespace
from model import Purifier_Classifier, set_random_norm_mixed

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluation_robustness(model, classifier, dataloader, attacks, config):
    atk_acc_dict = {}
    acc_dict = {
                "test_nat": 0,
                "test_adv": 0,
                "test_cln": 0,
                "test_rob": 0,
                "test_rob_gray": 0,
                }
    
    wb_atks = {}
    gb_atks = {}
    
    ### Attack ###
    for atk in attacks:
        if atk == "pgd":
            atk_acc_dict[atk] = acc_dict.copy()
            wb_atks[atk] = torchattacks.PGD(model, eps=config.TEST.EPS, alpha=2/255, steps=config.TEST.PGD_STEP)
            gb_atks[atk] = torchattacks.PGD(classifier, eps=config.TEST.EPS, alpha=2/255, steps=config.TEST.PGD_STEP)
        elif atk == "eot-pgd":
            atk_acc_dict[atk] = acc_dict.copy()
            wb_atks[atk] = torchattacks.EOTPGD(model, eps=config.TEST.EPS, alpha=2/255, 
                                               steps=config.TEST.PGD_STEP, eot_iter=config.TEST.EOT_ITER, random_start=True)
            gb_atks[atk] = torchattacks.EOTPGD(classifier, eps=config.TEST.EPS, alpha=2/255, 
                                               steps=config.TEST.PGD_STEP, eot_iter=config.TEST.EOT_ITER, random_start=True)
        elif atk == "fgsm":
            atk_acc_dict[atk] = acc_dict.copy()
            wb_atks[atk] = torchattacks.FGSM(model, eps=config.TEST.EPS)
            gb_atks[atk] = torchattacks.FGSM(classifier, eps=config.TEST.EPS)
        elif atk == "cw":
            atk_acc_dict[atk] = acc_dict.copy()
            wb_atks[atk] = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
            gb_atks[atk] = torchattacks.CW(classifier, c=1, kappa=0, steps=100, lr=0.01)
        elif atk == "aa":
            atk_acc_dict[atk] = acc_dict.copy()
            wb_atks[atk] = torchattacks.AutoAttack(model, norm='Linf', eps=config.TEST.EPS, 
                                                   version='standard', n_classes=config.NUM_CLASS, seed=config.TEST.SEED, verbose=False)
            gb_atks[atk] = torchattacks.AutoAttack(classifier, norm='Linf', eps=config.TEST.EPS, 
                                                   version='standard', n_classes=config.NUM_CLASS, seed=config.TEST.SEED, verbose=False)
        else:
            raise Exception("Attack method is wrong. It should be pgd or aa.")
    
    total_test = 0
    model.eval()
    classifier.eval()
    
    for idx, (x_test, y_test) in enumerate(tqdm(dataloader)):
        x_test, y_test = x_test.to(device), y_test.to(device)
        
        set_random_norm_mixed(model, config.BLOCK_WISE)
        
        for (wb_atk_name, wb_atk), (gb_atk_name, gb_atk) in zip(wb_atks.items(), gb_atks.items()):
            wb_adv = wb_atk(x_test, y_test)
            gb_adv = gb_atk(x_test, y_test)
        
            with torch.no_grad():
                ## Iterative forward
                pred_cln = torch.zeros((x_test.size(0), config.NUM_CLASS)).to(device)
                pred_rob = torch.zeros((x_test.size(0), config.NUM_CLASS)).to(device)
                pred_rob_gray = torch.zeros((x_test.size(0), config.NUM_CLASS)).to(device)
                pred_nat = classifier(x_test)
                pred_adv = classifier(gb_adv)
                for _ in range(config.NUM_ITER):
                    set_random_norm_mixed(model, config.BLOCK_WISE)
                    pred_cln_temp, x_cln = model(x_test, True)
                    pred_cln += pred_cln_temp
                    pred_rob_temp, x_pur = model(wb_adv, True)
                    pred_rob += pred_rob_temp
                    pred_rob_gray_temp, x_pur_gray = model(gb_adv, True)
                    pred_rob_gray += pred_rob_temp
                    
                prediction_dict = { 
                    "test_nat": pred_nat,
                    "test_adv": pred_adv,
                    "test_cln": pred_cln / config.NUM_ITER,
                    "test_rob": pred_rob / config.NUM_ITER,
                    "test_rob_gray": pred_rob_gray / config.NUM_ITER,
                    }
                
                for k, v in prediction_dict.items():
                    _, y_pred = torch.max(v.data, 1)
                    atk_acc_dict[wb_atk_name][k] += (y_pred == y_test).sum().item()
            
        total_test += x_test.size(0)
    
    for atk, acc in atk_acc_dict.items():
        for k in atk_acc_dict[atk].keys():
            atk_acc_dict[atk][k] = (atk_acc_dict[atk][k] / total_test) * 100
        
    images = {
        "raw": x_test,
        "clean": x_cln,
        "adv": wb_adv,
        "pur": x_pur,
        }
    
    return atk_acc_dict, images

def main():
    parser = argparse.ArgumentParser()
    
    ### Experiment configuration ###
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    print(args)
    
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
    set_manual_seed(config.TEST.SEED)
    
    ### Result path ###
    if not os.path.exists(config.RESULT_PATH):
        os.makedirs(config.RESULT_PATH)
    result_path = os.path.join(config.RESULT_PATH, config.PROJECT, config.EXP)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    log_path = os.path.join(result_path, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    norm_names = ""
    for norm in config.NORM_LIST:
        norm_names += f"{norm}_" 
        
    conv_names = ""
    for conv in config.CONV_LIST:
        conv_names += f"{conv}_"
    ckpt_path = os.path.join(result_path, f"{config.DATASET}_{norm_names}{conv_names}final.pth") if config.TEST.CKPT_PATH is None else config.TEST.CKPT_PATH
    
    model = Purifier_Classifier(config).to(device)
    classifier = get_classifier(config).to(device)
    
    checkpoint = torch.load(ckpt_path)
    model.purifier.load_state_dict(checkpoint["purifier"])
    logging.info("Load the model at step %d" % checkpoint["iteration"])
    
    logging.info("")
    logging.info("[!] Evaluation ...")
    test_loader = get_dataloader(dataset_name=config.DATASET, which="val", 
                                    subsample=config.TEST.TEST_SUBSAMPLE,
                                    batch_size=config.TEST.BATCH_SIZE, config=config)
    
    atk_acc_dict, _ = evaluation_robustness(model, classifier, test_loader, config.TEST.ATTACKS, config)
    
    for atk_name, acc_dict in atk_acc_dict.items():
        logging.info("")
        logging.info(colors.YELLOW + f"[Evaluation results on testset of {config.DATASET}]" + colors.WHITE)
        print_accuracies = {"Attack":              atk_name,
                            "Classifier":          config.CLASSIFIER,
                            }
        print_accuracies = {**print_accuracies, **acc_dict}
        
        with open(os.path.join(log_path, f"{config.DATASET}_{config.CLASSIFIER}.txt"), "a") as f:
            f.write(f"\n\n Date: {time.strftime('%y%m%d_%H%M%S')}")
            f.write(f"\nWeight: {ckpt_path}")
            f.write(f"\nSeed: {config.TEST.SEED}")
            for name, acc in print_accuracies.items():
                if type(acc) == float:
                    logging.info(f"{name:>20s} | {acc:05.2f}%")
                    f.write(f"\n{name:>20s} | {acc:05.2f}%")
                else:
                    logging.info(f"{name:>20s} | {acc}")
                    f.write(f"\n{name:>20s} | {acc}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()