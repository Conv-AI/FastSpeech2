import argparse
import os
import wandb
from datetime import datetime
from accelerate import Accelerator
import time
import torch
import torch.distributed as dist
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs
    batch_size = train_config["optimizer"]["batch_size"]
    
#     wandb_project_name = train_config["logging"]["project"]
#     wandb_run_name = str(train_config["logging"]["run"])+"_"+str(batch_size)+'_'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     wandb.init(project=wandb_project_name, name=wandb_run_name, config=configs[0])

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
                        )
   
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    assert batch_size * group_size < len(val_dataset)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    
    print("Len loader:"+str(len(loader)))
    accelerator = Accelerator(log_with="wandb", project_dir=train_config["path"]["ckpt_path"])
    device = accelerator.device
    
    wandb_project_name = train_config["logging"]["project"]
    wandb_run_name = str(train_config["logging"]["run"])+"_"+str(batch_size)+'_'+datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=wandb_project_name, name=wandb_run_name, config=configs[0])
    
    accelerator.init_trackers(
    project_name=wandb_project_name,
    config=configs[0],
    init_kwargs={"wandb": {"entity": "convai-tts"}}
    )


    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model.to(device)
    
#     ckpt_file_path = './checkpoint.pth'
#     checkpoint = torch.load(ckpt_file_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['model'])
    
    #model = nn.DataParallel(model)
    #model = nn.parallel.DistributedDataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    

    # Load vocoder
#     vocoder = get_vocoder(model_config, device)
    vocoder = None

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    while True:
        fail_count = 0
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            #print("Len batches:"+str(len(batchs)))
            for batch in batchs:
                #print("Len batch:"+str(len(batch)))
                
                batch = to_device(batch, device)
                #print("Len batches:"+str(len(batchs)))
                # Forward
                #try:
                output = model(*(batch[2:]))
#                 except:
#                     print(batch)
#                     fail_count += 1
#                     print(fail_count)
#                     continue

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                accelerator.backward(total_loss)
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )
                    message2_ = {"total_loss": losses[0], "mel loss": losses[1], "postnet loss": losses[2],
                                "pitch loss": losses[3], "energy loss": losses[4], "duration loss": losses[5]}
                    accelerator.log(message2_)


                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, logger_stage="training")

#                 if step % val_step == 0:
#                     print("running validation...")
#                     torch.cuda.empty_cache()
#                     model.eval()
                    
#                     loss_sums = [0 for _ in range(6)]
                    
#                     for batchs in tqdm(val_loader):
#                         for batch in batchs:
#                             batch = to_device(batch, device)
#                             with torch.no_grad():
#                                 # Forward
#                                 output = model(*(batch[2:]))

#                                 # Cal Loss
#                                 losses = Loss(batch, output)
#                                 torch.cuda.empty_cache()

#                                 for i in range(len(losses)):
#                                     loss_sums[i] += losses[i].item() * len(batch[0])

#                     loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
                    
#                     message = {"total_loss": loss_means[0], "mel loss": loss_means[1], "postnet loss": loss_means[2],
#                  "pitch loss": loss_means[3], "energy loss": loss_means[4], "duration loss": loss_means[5]}
#                     accelerator.log(message)

                    
#                     #message = evaluate(model, step, configs, val_logger, vocoder, accelerator, device)
# #                     with open(os.path.join(val_log_path, "log.txt"), "a") as f:
# #                         f.write(message + "\n")
#                     outer_bar.write(message)
#                     accelerator.log(message)

#                     model.train()

                if step % save_step == 0:
                    print("saving checkpoint...")
                    accelerator.register_for_checkpointing(scheduler)

                    accelerator.save_state()
                
                    model.to(device)
#                     torch.save(
#                         {
#                             "model": model.module.state_dict(),
#                             "optimizer": optimizer._optimizer.state_dict(),
#                         },
#                         os.path.join(
#                             train_config["path"]["ckpt_path"],
#                             "{}-{}.pth.tar".format(step, time.time()),
#                         ),
#                     )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)
                
                #torch.cuda.empty_cache()

            inner_bar.update(1)
        epoch += 1
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    
    batch_size = 32*8
    
    

    main(args, configs)
