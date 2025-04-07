import torch
checkpoint = torch.load('./Checkpoints_PSM_TTT_24/finetuned_checkpoint-10.pt')
print(checkpoint['step'])