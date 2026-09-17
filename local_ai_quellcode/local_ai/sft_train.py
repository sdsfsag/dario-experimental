import argparse
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig, TrainConfig
from data import SFTDataset
from hardware import select_device
from model import Transformer
from tokenizer import BPETokenizer
from train import Trainer, load_checkpoint, seed_all


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--pretrained', help='Your own pretraining checkpoint')
    source.add_argument('--resume')
    p.add_argument('--train-data', default='prepared/sft_train')
    p.add_argument('--val-data', default='prepared/sft_val')
    p.add_argument('--output')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--device', default='auto')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args()
    a.output = a.output or (str(Path(a.resume).parent) if a.resume else 'runs/sft')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    payload = load_checkpoint(a.resume or a.pretrained)
    if not payload['overfit'] or not payload['overfit']['passed']:
        raise ValueError('Checkpoint has no successful overfit test')
    if a.pretrained and payload['stage'] != 'pretrain':
        raise ValueError('--pretrained must be your own pretraining checkpoint')
    c = ModelConfig(**payload['model_config'])
    t = TrainConfig(**payload['training_config']) if a.resume else TrainConfig(
        steps=a.steps, batch_size=a.batch_size, grad_accum=a.grad_accum, lr=a.lr,
        warmup_steps=max(1, int(a.steps*0.05)), weight_decay=0.01)
    tokenizer = BPETokenizer(Tokenizer.from_str(payload['tokenizer_json']))
    train_data = SFTDataset(a.train_data, c.context_length, tokenizer)
    val_data = SFTDataset(a.val_data, c.context_length, tokenizer)
    if set(train_data.meta['conversation_hashes']) & set(val_data.meta['conversation_hashes']):
        raise ValueError('Train/validation conversations overlap')
    if not a.resume and (Path(a.output)/'last.pt').exists():
        raise FileExistsError('Use --resume or a new --output directory')
    seed_all(t.seed)
    device = select_device(a.device)
    model = Transformer(c).to(device)
    model.load_state_dict(payload['model'])
    Trainer(model, t, tokenizer, train_data, val_data, a.output, device, stage='sft',
            resume=payload if a.resume else None, overfit=payload['overfit']).run(a.stop_after)


if __name__ == '__main__':
    main()
