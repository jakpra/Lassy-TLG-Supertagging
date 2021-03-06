import sys

import time

from Transformers.Transformer import Transformer
from Transformers.utils import FuzzyLoss, CustomLRScheduler, noam_scheme, make_mask as Mask, Scheduler, make_cosine_schedule

import torch
from torch import nn
from torch.nn import functional as F
import torch.cuda as cuda
from src.dataprep import TLGDataset
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from collections import Counter

import numpy as np

from src import dataprep

from typing import List, Any, Tuple, Union, Callable

FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]


PAD = '<PADDING>'
START = '<START>'
SEP = '<SEP>'


def accuracy(predictions: LongTensor, truth: LongTensor, ignore_idx: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    correct_words = torch.ones(predictions.size())
    correct_words[predictions != truth] = 0
    correct_words[truth == ignore_idx] = 1

    correct_sentences = correct_words.prod(dim=1)
    num_correct_sentences = correct_sentences.sum().item()

    num_correct_words = correct_words.sum().item()
    num_masked_words = len(truth[truth == ignore_idx])

    return (predictions.shape[0], num_correct_sentences), \
           (predictions.shape[0] * predictions.shape[1] - num_masked_words, num_correct_words - num_masked_words)


class Supertagger(nn.Module):
    def __init__(self, num_classes: int, encoder_heads: int, decoder_heads: int, encoder_layers: int,
                 decoder_layers: int, d_intermediate: int, device: str, dropout: float=0.1, d_model: int = 300,
                 padding_index=0) -> None:
        super(Supertagger, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(num_classes=num_classes, encoder_heads=encoder_heads,
                                       decoder_heads=decoder_heads, encoder_layers=encoder_layers,
                                       decoder_layers=decoder_layers, d_model=d_model, d_intermediate=d_intermediate,
                                       dropout=dropout, device=device, reuse_embedding=True, padding_index=padding_index)
        self.device = device

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor, encoder_mask: LongTensor,
                decoder_mask: FloatTensor) -> FloatTensor:
        return self.transformer.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

    def train_epoch(self, dataset: TLGDataset, batch_size: int,
                    criterion: Callable[[FloatTensor, LongTensor], FloatTensor],
                    optimizer: optim.Optimizer, train_indices: List[int], n_print=100, epoch=0) -> Tuple[float, int, int, int, int]:
        self.train()

        permutation = np.random.permutation(train_indices)

        batch_start = 0
        loss = 0.
        BS, BTS, BW, BTW = 0, 0, 0, 0

        running_batch_time = 0.0

        # while batch_start < len(permutation):
        for i in range(len(permutation)):
            start_time = time.time()
            optimizer.zero_grad()
            # batch_end = min([batch_start + batch_size, len(permutation)])

            # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
            # batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]
            batch_x = dataset.X[permutation[i]]
            batch_y = dataset.Y[permutation[i]]

            # lens = list(map(len, batch_x))
            lens = torch.sum((batch_x.word != dataset.x_pad_token).long(), dim=1).to(self.device)

            # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
            # batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)
            batch_e = F.embedding(batch_y.to(self.device), self.transformer.embedding_matrix)

            encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_x.shape[1])
            for i, l in enumerate(lens):
                encoder_mask[i, :, l::] = 0
            encoder_mask = encoder_mask.to(self.device)
            decoder_mask = Mask((batch_x.shape[0], batch_y.shape[1], batch_y.shape[1])).to(self.device)  # does this have to be t()?

            batch_p = self.forward(batch_x, batch_e, encoder_mask, decoder_mask)

            batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_y[:, 1:].to(self.device)) / lens.float().sum()
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            argmaxes = batch_p.argmax(dim=-1)
            # print('pre argmaxes', argmaxes.size(), argmaxes[0])
            # print('pre y', batch_y.size(), batch_y[0])
            argmaxes = argmaxes[:, :-1]
            y = batch_y[:, 1:]
            # print('post argmaxes', argmaxes.size(), argmaxes[0])
            # print('post y', y.size(), y[0])

            (bs, bts), (bw, btw) = accuracy(argmaxes, y.to(self.device), dataset.type_dict[PAD])
            # (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_y[:, 1:], dataset.type_dict[PAD])
            BS += bs
            BTS += bts
            BW += bw
            BTW += btw

            running_batch_time += time.time() - start_time

            if i % n_print == n_print - 1:  # print every n mini-batches

                batch_time = running_batch_time / n_print
                print('[%d, %5d] loss: %.3f | acc: %.3f | %.1f %s | %.1f %s' % (epoch + 1, i + 1,
                                                                      loss / n_print,
                                                                      BTW / BW,
                                                                      batch_time if batch_time >= 1 else 1 / batch_time,
                                                                      's/batch' if batch_time >= 1 else 'batch(es)/s',
                                                                      batch_time/batch_size if batch_time/batch_size >= 1 else batch_size/batch_time,
                                                                      's/expl' if batch_time/batch_size >= 1 else 'expl(s)/s'),
                      file=sys.stderr)
                # if str(device).startswith('cuda'):
                #     print(torch.cuda.memory_summary(abbreviated=False), file=sys.stderr)

                # assist.info['batch'] = train_i + 1
                # assist.info['batch_loss'] = running_loss / n_print
                # assist.info['batch_acc'] = running_acc / n_print
                # assist.info['ex_per_s'] = batch_size / batch_time
                # assist.step()

                # running_loss = 0.0
                # running_acc = 0.0
                running_batch_time = 0.0

            # batch_start += batch_size

        return loss, BS, BTS, BW, BTW

    def eval_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int], gen,
                   criterion: Callable[[FloatTensor, LongTensor], FloatTensor]) -> Tuple[float, int, int, int, int]:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            loss = 0.
            BS, BTS, BW, BTW, BC, BTC = 0, 0, 0, 0, 0, 0
            # n_words = 0

            gold_categories, generated_categories, correct_categories = Counter(), Counter(), Counter()

            # while batch_start < len(permutation):
            for i in range(len(permutation)):
                # batch_end = min([batch_start + batch_size, len(permutation)])

                # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                # batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = dataset.X[permutation[i]]
                batch_y = dataset.Y[permutation[i]].long().to(self.device)
                # print(batch_y)

                # lens = list(map(len, batch_x))
                lens = torch.sum((batch_x.word != dataset.x_pad_token).long(), dim=1).to(self.device)

                # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
                # batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)

                encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_x.shape[1])
                for i, l in enumerate(lens):
                    encoder_mask[i, :, l::] = 0
                encoder_mask = encoder_mask.to(self.device)
                batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict[START])  # TODO: SEP?
                                                 # dataset.type_dict[SEP], lens)
                if batch_p.size(1) < batch_y.shape[1]:
                    batch_p = torch.cat([batch_p,
                                         torch.zeros(batch_p.shape[0], batch_y.shape[1] - batch_p.size(1), batch_p.shape[2]).to(batch_p)],
                                        dim=1)
                batch_loss = criterion(torch.log(batch_p[:, :-1]).permute(0, 2, 1), batch_y[:, 1:].to(self.device)) / lens.float().sum()
                loss += batch_loss.item()
                argmaxes = batch_p[:, :-1].argmax(dim=-1)
                y = batch_y[:, 1:]
                (bs, bts), (bw, btw) = accuracy(argmaxes, y.to(self.device), dataset.type_dict[PAD])
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                categories_gold = gen.extract_outputs(y)
                categories_hat = gen.extract_outputs(argmaxes)
                print('y', y[0].tolist(), file=sys.stderr)
                print('argmaxes', argmaxes[0].tolist(), file=sys.stderr)
                for b, (sequence, sequence_gold) in enumerate(zip(categories_hat, categories_gold)):
                    # print(sequence, sequence_gold)
                    for s, cat_gold in enumerate(sequence_gold):
                        if str(cat_gold) == PAD:
                            continue
                        cat = sequence[s] if s < len(sequence) else None
                        BC += 1
                        correct = False
                        # correct_index = [b, s] in correct_indices
                        # assert (cat == cat_gold) == (correct_index), (
                        #     b, s, cat, cat_gold, argmaxes[b, s], mask[b, s], y[b, s],
                        #     f'[{b}, {s}] {"not " if not correct_index else ""}in correct_indices')
                        if cat is None:
                            cat = 'None'
                        else:
                            try:
                                msg = cat.validate()
                            except RecursionError:
                                msg = 'Maximum recursion depth reached'
                                cat = msg
                            except:
                                raise
                            else:
                                if msg == 0:
                                    correct = cat_gold.equals(cat)
                                else:
                                    if hasattr(gen, 'max_depth') and cat.depth() >= gen.max_depth:
                                        msg = 'Max depth reached'
                                        # print(b, s, msg, str(cat), cat.s_expr())
                                        # cat = msg
                                    # elif hasattr(gen, 'max_len') and argmaxes.size(1) >= gen.max_len:
                                    #     msg = 'Max length reached'
                                        # print(b, s, msg, str(cat), cat.s_expr())
                                        # cat = msg
                                    else:
                                        # print(b, s, msg[0], str(cat), cat.s_expr(), file=sys.stderr)
                                        # print(argmaxes[b, s], file=sys.stderr)
                                        # cat = msg[0]
                                        msg = msg[0]
                                    cat = f'{cat} ({msg})'
                        gold_categories[str(cat_gold)] += 1
                        generated_categories[str(cat)] += 1
                        if correct:
                            BTC += 1
                            correct_categories[str(cat)] += 1

                # batch_start += batch_size

        return loss, BS, BTS, BW, BTW, BC, BTC, gold_categories, generated_categories, correct_categories

    # def eval_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int],
    #                criterion: Callable[[FloatTensor, LongTensor], FloatTensor]) -> Tuple[float, int, int, int, int]:
    #     self.eval()
    #
    #     with torch.no_grad():
    #
    #         permutation = val_indices
    #
    #         batch_start = 0
    #         loss = 0.
    #         BS, BTS, BW, BTW = 0, 0, 0, 0
    #
    #         # while batch_start < len(permutation):
    #         for i in range(len(permutation)):
    #         #     batch_end = min([batch_start + batch_size, len(permutation)])
    #
    #             # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
    #             # batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]
    #             batch_x = dataset.X[permutation[i]]  # .to(self.device)
    #             batch_y = dataset.Y[permutation[i]].long().to(self.device)
    #
    #             # lens = list(map(len, batch_x))
    #             lens = torch.sum((batch_x.word != dataset.x_pad_token).long(), dim=1).to(self.device)
    #
    #             # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
    #             # batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)
    #
    #             encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_x.shape[1])
    #             for i, l in enumerate(lens):
    #                 encoder_mask[i, :, l::] = 0
    #             encoder_mask = encoder_mask.to(self.device)
    #             batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict[START])
    #             batch_loss = criterion(torch.log(batch_p[:, :-1]).permute(0, 2, 1), batch_y[:, 1:])
    #             loss += batch_loss.item()
    #             (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_y[:, 1:], dataset.type_dict[PAD])
    #             BS += bs
    #             BTS += bts
    #             BW += bw
    #             BTW += btw
    #
    #             batch_start += batch_size
    #
    #     return loss, BS, BTS, BW, BTW

    def infer_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int], max_len: int) \
            -> List[List[int]]:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0

            P = []

            running_batch_time = 0.0

            # while batch_start < len(permutation):
            for i in range(len(permutation)):
                # batch_end = min([batch_start + batch_size, len(permutation)])

                # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = dataset.X[permutation[i]]
                # batch_y = dataset.Y[permutation[i]]  # TODO: is truncating the output to the gold standard fair game?

                # lens = list(map(len, batch_x))
                lens = torch.sum((batch_x.word != dataset.x_pad_token).long(), dim=1).to(self.device)

                # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)

                # print('x shape, size', batch_x.shape, batch_x.size())

                encoder_mask = torch.ones(batch_x.shape[0], max_len * batch_x.shape[1], batch_x.shape[1])
                # encoder_mask = torch.ones(batch_x.shape[0], batch_y.shape[1], batch_x.shape[1])
                for i, l in enumerate(lens):
                    encoder_mask[i, :, l::] = 0
                encoder_mask = encoder_mask.to(self.device)

                start_time = time.time()
                batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict[START])
                                                 # dataset.type_dict[SEP], lens)
                running_batch_time += time.time() - start_time

                batch_p = batch_p[:, :-1].argmax(dim=-1).cpu().numpy().tolist()
                # P.append(batch_p)
                P.extend(batch_p)
                # batch_start += batch_size

        return P, running_batch_time

    def eval_epoch_beam(self, dataset: TLGDataset, batch_size: int, val_indices: List[int], beam_width: int) -> Any:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            BS, BTS, BW, BTW = 0, 0, 0, 0

            # while batch_start < len(permutation):
            for i in range(len(permutation)):
                # batch_end = min([batch_start + batch_size, len(permutation)])

                # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                # batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = dataset.X[permutation[i]]
                batch_y = dataset.Y[permutation[i]]

                lens = list(map(len, batch_x))

                # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
                # batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)

                encoder_mask = torch.ones(batch_x.shape[0], batch_y.shape[1], batch_x.shape[1])
                for i, l in enumerate(lens):
                    l = l - 1
                    encoder_mask[i, :, l::] = 0
                encoder_mask = encoder_mask.to(self.device)
                paths, scores = self.transformer.vectorized_beam_search(batch_x, encoder_mask, dataset.type_dict[START],
                                                                        beam_width=beam_width)
                # todo: placeholder--take best beam as the only beam
                batch_p = paths[0]
                (bs, bts), (bw, btw) = accuracy(batch_p, batch_y[:, 1:], dataset.type_dict[PAD])
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                # batch_start += batch_size

        return BS, BTS, BW, BTW


def bpe_ft():
    tlg = dataprep.bpe_ft()

    d_model = 300
    batch_size = 64
    num_epochs = 1000

    num_classes = len(tlg.type_dict) + 1
    n = Supertagger(num_classes=num_classes, encoder_heads=3, decoder_heads=3, encoder_layers=4, decoder_layers=4,
                    d_intermediate=d_model, device='cuda', dropout=0.2, d_model=d_model)

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.1)
    a = optim.Adam(n.parameters(), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-04)
    o = CustomLRScheduler(a, [noam_scheme], d_model=d_model, warmup_steps=4000, batch_size=4*batch_size)

    splitpoints = list(map(int, [np.floor(0.8 * len(tlg.X)), np.floor(0.9 * len(tlg.X))]))
    indices = list(range(len(tlg.X)))
    np.random.shuffle(indices)
    train_indices = indices[:splitpoints[0]]
    val_indices = sorted(indices[splitpoints[0]:splitpoints[1]], key=lambda x: len(tlg.Y[x]))
    test_indices = sorted(indices[splitpoints[1]:], key=lambda x: len(tlg.Y[x]))

    for i in range(num_epochs):
        loss, bs, bts, bw, btw = n.train_epoch(tlg, batch_size, L, o, train_indices)
        val_bs, val_bts, val_bw, val_btw = n.eval_epoch(tlg, batch_size, val_indices)
        print('Epoch {}'.format(i))
        print(' Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(loss, bts / bs, btw / bw))
        print(' (Validation) Sentence Accuracy: {}, Word Accuracy: {}'.format(val_bts / val_bs, val_btw / val_bw))


def do_everything(tlg=None):
    from pathlib import Path
    from Transformers.utils import EncoderInput

    from ccg.parser.evaluation.evaluation import Evaluator
    from ccg.util.reader import AUTODerivationsReader, ASTDerivationsReader, StaggedDerivationsReader
    from ccg.representation.category import Category
    from ccg.representation.derivation import Derivation
    from ccg.util.mode import Mode
    import ccg.util.argparse as ap

    args = ap.main()

    if not args.model_exists:
        assert args.mode == Mode.train, 'If model does not exist, mode has to be \'train\'.'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda and cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    print('Device selected:', args.device, file=sys.stderr)

    # pretrained_path = 'type_LM'
    #
    # split_path = 'split.p'

    d_model = args.hidden_dims[0] if args.hidden_dims else 768
    dropout = args.dropout[0] if args.dropout else 0.2
    batch_size = args.batch_size
    beam_size = 3

    # 3,4,4,128,0.1,0.1,4000
    if tlg is None:
        # tlg = DataPrep.do_everything_elmo(model_path
        #                     ='/home/kokos/Documents/Projects/Lassy/LassySupertagging/ELMoForManyLangs/Models/English',
        #                     data_file='data/XYZ_ccg.p')
        # tlg = dataprep.do_everything()
        tlg, train_indices, val_indices, test_indices, st = dataprep.do_everything_ccg(args, d_model)

    gen = st.generators[0]
    # num_classes = len(tlg.type_dict) + 1
    num_classes = gen.output_dim + 1

    # logfile = open(f'{args.model}.log', 'w')


    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super(EncoderWrapper, self).__init__()
            self.encoder = enc

        def forward(self, x):
            return EncoderInput(encoder_input=self.encoder(vars(x.encoder_input), word_mask=x.mask, device=args.device)[0],
                                mask=x.mask)

    if args.mode == Mode.train:

        # print('Training on {} classes'.format(len(tlg.type_dict)))
        print('Training on {} classes'.format(gen.output_dim), file=sys.stderr)
        # print('Training on {} classes'.format(gen.output_dim), file=logfile)
        # n = Supertagger(num_classes, 4, 3, 3, 600, dropout=0.2, device='cuda', d_model=d_model)
        model = Supertagger(num_classes + 1, encoder_heads=3, decoder_heads=8, encoder_layers=1,
                            decoder_layers=2, d_intermediate=d_model, device=args.device, dropout=dropout,
                            d_model=d_model,
                            padding_index=gen.out_to_ix[PAD])

        model.transformer.encoder = EncoderWrapper(st.span_encoder)

        # L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.2, gen.out_to_ix[PAD])
        L = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=gen.out_to_ix[PAD])

        print(model, file=sys.stderr)

        if args.model_exists:
            print('Found model. Loading parameters...', file=sys.stderr)
            with open(f'{args.model}.pt', 'rb') as f:
                # self_dict = n.state_dict()
                # import re
                checkpoint = torch.load(f, map_location=args.device)
                # for k, p in pretrained.items():
                # TODO: I assume these are for a model that was pretrained with a different architecture (TypeLM)?
                # k = re.sub(r'network', 'transformer.decoder', k)
                # k = re.sub(r'mha', 'mask_mha', k)
                # k = re.sub(r'embedding_matrix', 'transformer.embedding_matrix', k)
                # k = re.sub(r'predictor', 'transformer.predictor', k)
                # if k in self_dict.keys():
                #     self_dict[k] = p
                #     print('replaced {}'.format(k))
                # else:
                #     continue
                # n.load_state_dict(self_dict)
            model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            best_val = checkpoint.get('dev_acc', 0.0)
            best_atom_val = checkpoint.get('dev_atom_acc', 0.0)
            best_val_loss = checkpoint.get('dev_loss', None)
            start_epoch = checkpoint.get('epoch', 0)

            if 'model_state_dict' in checkpoint:
                del checkpoint['model_state_dict']
            else:
                del checkpoint

            print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
            print(sum(p.numel() for p in model.transformer.encoder.parameters() if p.requires_grad),
                  ' parameters in encoder',
                  file=sys.stderr)
            print(sum(p.numel() for p in model.transformer.decoder.parameters() if p.requires_grad),
                  ' parameters in decoder',
                  file=sys.stderr)

            print('best epoch:', start_epoch, file=sys.stderr)
            print('best dev acc:', best_val, file=sys.stderr)
            print('best dev atomic acc:', best_atom_val, file=sys.stderr)
            print('best dev loss:', best_val_loss, file=sys.stderr)

            model = model.to(args.device)
            # model.eval()

            dev_loss, dev_bs, dev_bts, dev_bw, dev_btw, dev_bc, dev_btc, gold_categories, generated_categories, correct_categories = model.eval_epoch(tlg, batch_size, val_indices, gen, L)
            dev_atom_acc = dev_btw / dev_bw
            dev_cat_acc = dev_btc / dev_bc
            # print('Epoch {}'.format(start_epoch), file=sys.stderr)
            # print(' VALIDATION Loss: {}, Sentence Accuracy: {}, Atomic Accuracy: {}, Category Accuracy: {}'.format(
            #     loss, bts / bs, btw / bw, cat_acc), file=sys.stderr)
            print(
                '[epoch %d summary] train loss: %.3f | train atom acc: %.3f | dev loss: %.3f | dev atom acc: %.3f | dev acc: %.3f' %
                (start_epoch,
                 0.0,
                 0.0,
                 dev_loss,
                 dev_atom_acc,
                 dev_cat_acc),
                file=sys.stderr)

            print(f'most common gold categories (out of {dev_bc} in dev): '
                  f'{" | ".join(str(item) for item in gold_categories.most_common(10))}', file=sys.stderr)
            print(f'most common generated categories (out of {dev_bc} in dev): '
                  f'{" | ".join(str(item) for item in generated_categories.most_common(10))}', file=sys.stderr)
            print(f'most common correct categories (out of {dev_bc} in dev): '
                  f'{" | ".join(str(item) for item in correct_categories.most_common(10))}', file=sys.stderr)

        else:
            best_val = 0.0
            best_val_loss = None
            start_epoch = 0

            print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
            print(sum(p.numel() for p in model.transformer.encoder.parameters() if p.requires_grad),
                  ' parameters in encoder',
                  file=sys.stderr)
            print(sum(p.numel() for p in model.transformer.decoder.parameters() if p.requires_grad),
                  ' parameters in decoder',
                  file=sys.stderr)


        epochs = args.epochs
        # TODO: does reloading work at all with the LRScheduler?

        if args.model_exists:
            print('Resuming training...', file=sys.stderr)
        else:
            print('Model not found. Starting from scratch...', file=sys.stderr)

        assert(all(list(map(lambda x: x.requires_grad, model.parameters()))))

        param_groups = [{'params': model.transformer.decoder.parameters(), 'lr': args.learning_rate}]
        if args.span_encoder in ('bert', 'roberta', 'albert'):
            param_groups.append({'params': model.transformer.encoder.parameters(), 'lr': args.bert_learning_rate})

        # a = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-04)
        a = optim.AdamW(param_groups, eps=args.epsilon, weight_decay=args.decay)

        def var_rate(rate):
            # return lambda _step, d_model, warmup_steps, batch_size=2048: \
            return lambda _step, d_model, warmup_steps, **kwargs: \
                noam_scheme(_step=_step, d_model=d_model, warmup_steps=warmup_steps, batch_size=batch_size/rate)

        # o = CustomLRScheduler(a, [var_rate(args.learning_rate), var_rate(args.bert_learning_rate)],
        #                       d_model=d_model, warmup_steps=4000, batch_size=batch_size/args.learning_rate)

        steps = len(tlg)*epochs
        warmup = steps//10
        o = Scheduler(a, make_cosine_schedule(1, warmup, steps - warmup), (args.learning_rate, args.bert_learning_rate))

        # with open(split_path, 'rb') as f:
        #     train_indices, val_indices, test_indices = pickle.load(f)

        # best_val = 0.5

        for i in range(start_epoch, start_epoch+epochs):
            # loss, bs, bts, bw, btw = model.train_epoch(tlg, batch_size, L, o, train_indices)

            # model.train()
            loss, bs, bts, bw, btw = model.train_epoch(tlg, batch_size, L, o if args.use_schedule else a, train_indices)
            # print('Epoch {}'.format(i+1), file=sys.stderr)
            # print(' Loss: {}, Sentence Accuracy: {}, Atomic Accuracy: {}'.format(loss, bts/bs, btw/bw), file=sys.stderr)
            # print('Epoch {}'.format(i + 1), file=logfile)
            # print(' Loss: {}, Sentence Accuracy: {}, Atomic Accuracy: {}'.format(loss, bts / bs, btw / bw),
            #       file=logfile)
            epoch_acc = btw / bw

            # if i % 5 == 0 and i != 0:
            # if i % args.n_print == args.n_print - 1 and i != 0:
            # model.eval()
            dev_loss, dev_bs, dev_bts, dev_bw, dev_btw, dev_bc, dev_btc, gold_categories, generated_categories, correct_categories = model.eval_epoch(
                tlg, batch_size, val_indices, gen, L)
            dev_atom_acc = dev_btw / dev_bw
            dev_cat_acc = dev_btc / dev_bc
            # print('Epoch {}'.format(i+1), file=sys.stderr)
            # print(' VALIDATION Loss: {}, Sentence Accuracy: {}, Atomic Accuracy: {}, Category Accuracy: {}'.format(
            #     loss, bts / bs, btw / bw, cat_acc), file=sys.stderr)
            #
            # print(f'most common gold categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in gold_categories.most_common(10))}', file=sys.stderr)
            # print(f'most common generated categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in generated_categories.most_common(10))}', file=sys.stderr)
            # print(f'most common correct categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in correct_categories.most_common(10))}', file=sys.stderr)
            #
            # print('Epoch {}'.format(i+1), file=logfile)
            # print(' VALIDATION Loss: {}, Sentence Accuracy: {}, Atomic Accuracy: {}, Category Accuracy: {}'.format(
            #     loss, bts / bs, btw / bw, cat_acc), file=logfile)
            #
            # print(f'most common gold categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in gold_categories.most_common(10))}', file=logfile)
            # print(f'most common generated categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in generated_categories.most_common(10))}', file=logfile)
            # print(f'most common correct categories (out of {bc} in dev): '
            #       f'{" | ".join(str(item) for item in correct_categories.most_common(10))}', file=logfile)

            # bs, bts, bw, btw = n.eval_epoch_beam(tlg, batch_size, val_indices, beam_size)
            # print(' BEAM VALIDATION Sentence Accuracy: {}, Word Accuracy: {}'.format(bts / bs, btw / bw))
            # if bts/bs > best_val:
            #     best_val = bts/bs
            if dev_cat_acc > best_val or dev_cat_acc == best_val and (best_val_loss is None or dev_loss < best_val_loss):
                best_epoch = i + 1
                best_val = dev_cat_acc
                best_val_loss = dev_loss
                checkpoint = {'model_state_dict': model.state_dict(),
                              'epoch': best_epoch,
                              'dev_acc': dev_cat_acc,
                              'dev_atom_acc': dev_atom_acc,
                              'dev_loss': dev_loss
                              }
                # with open('model_{}_{}.pt'.format(i, btw/bw), 'wb') as f:
                with open(f'{args.model}.pt', 'wb') as f:
                    torch.save(checkpoint, f)

            print('[epoch %d summary] train loss: %.3f | train atom acc: %.3f | dev loss: %.3f | dev atom acc: %.3f | dev acc: %.3f' %
                  (i + 1,
                   loss,
                   epoch_acc,
                   dev_loss,
                   dev_atom_acc,
                   dev_cat_acc),
                  file=sys.stderr)

            if len(args.development_files) > 0:
                print(f'most common gold categories (out of {dev_bc} in dev): '
                      f'{" | ".join(str(item) for item in gold_categories.most_common(10))}',
                      file=sys.stderr)
                print(f'most common generated categories (out of {dev_bc} in dev): '
                      f'{" | ".join(str(item) for item in generated_categories.most_common(10))}',
                      file=sys.stderr)
                print(f'most common correct categories (out of {dev_bc} in dev): '
                      f'{" | ".join(str(item) for item in correct_categories.most_common(10))}',
                      file=sys.stderr)

            sys.stderr.flush()

        del model
        torch.cuda.empty_cache()

    print('Found model. Loading parameters...', file=sys.stderr)
    # print('Found model. Loading parameters...', file=logfile)

    with open(f'{args.model}.pt', 'rb') as f:
        checkpoint = torch.load(f, map_location=args.device)

    model = Supertagger(num_classes + 1, encoder_heads=3, decoder_heads=8, encoder_layers=1,
                        decoder_layers=2, d_intermediate=d_model, device=args.device, dropout=dropout,
                        d_model=d_model,
                        padding_index=gen.out_to_ix[PAD])

    model.transformer.encoder = EncoderWrapper(st.span_encoder)


    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    best_val = checkpoint.get('dev_acc', 0.0)
    best_atom_val = checkpoint.get('dev_atom_acc', 0.0)
    best_val_loss = checkpoint.get('dev_loss', None)
    start_epoch = checkpoint.get('epoch', 0)

    if 'model_state_dict' in checkpoint:
        del checkpoint['model_state_dict']
    else:
        del checkpoint

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=sys.stderr)
    print(sum(p.numel() for p in model.transformer.encoder.parameters() if p.requires_grad), ' parameters in encoder',
          file=sys.stderr)
    print(sum(p.numel() for p in model.transformer.decoder.parameters() if p.requires_grad), ' parameters in decoder',
          file=sys.stderr)

    print('best epoch:', start_epoch, file=sys.stderr)
    print('best dev acc:', best_val, file=sys.stderr)
    print('best dev atomic acc:', best_atom_val, file=sys.stderr)
    print('best dev loss:', best_val_loss, file=sys.stderr)

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad), ' parameters', file=logfile)
    # print(sum(p.numel() for p in model.transformer.encoder.parameters() if p.requires_grad), ' parameters in encoder',
    #       file=logfile)
    # print(sum(p.numel() for p in model.transformer.decoder.parameters() if p.requires_grad), ' parameters in decoder',
    #       file=logfile)

    # print('best epoch:', start_epoch, file=logfile)
    # print('best dev acc:', best_val, file=logfile)
    # print('best dev atomic acc:', best_atom_val, file=logfile)
    # print('best dev loss:', best_val_loss, file=logfile)

    model = model.to(args.device)
    # model.eval()

    argmaxes, running_batch_time = model.infer_epoch(tlg, batch_size, test_indices, gen.max_len + 1)  # max_len argument is per-word, not for whole sequence!!!
    cats = gen.extract_outputs(argmaxes)
    # _, _, _, _, _, _, _, _, cats, _ = model.eval_epoch(
    #     tlg, batch_size, test_indices, gen, L)

    test_len = len(test_indices)
    print('[test summary] %.3f batches/s | %.3f expls/s' %
          (test_len / running_batch_time,
           (test_len * batch_size) / running_batch_time
           ),
          file=sys.stderr)
        # running_test_acc += correct_bool.float().mean(dim=[0, 1]).item()

    # dev_acc = running_test_acc / len(testloader)
    evl = Evaluator(args.training_files, max_depth=6)

    cats = iter(cats)

    testing_format = args.testing_format or args.format
    if testing_format == 'ast':
        dr = ASTDerivationsReader
    elif testing_format == 'stagged':
        dr = StaggedDerivationsReader
    else:
        dr = AUTODerivationsReader

    if args.out is None:
        if args.oracle_scoring:
            args.out = 'oracle.auto'
        else:
            args.out = f'{args.model}.auto'

    with open(args.out, 'w') as f:
        pass

    tab_out = f'{args.model}.tsv'
    with open(tab_out, 'w') as f:
        f.write(f'{args.model}\n')

    for filename in args.testing_files:
        ds = dr(filename)
        while True:
            try:
                deriv = ds.next()
            except StopIteration:
                break
            try:
                tags = next(cats)
            except StopIteration:
                break
            deriv, ID = deriv['DERIVATION'], deriv['ID']
            if len(tags) != len(deriv.sentence):
                # print(f'words {len(deriv.sentence)}  tags {len(tags)}', file=sys.stderr)
                # print(f'{deriv.sentence}\n{tags}', file=sys.stderr)
                # print('\n'.join([f'{w}\t{t}' for w, t in zip(tags, deriv.sentence)]), file=sys.stderr)
                # assert len(tags) == len(deriv.sentence)
                tags.extend(len(deriv.sentence) * [Category(PAD)])
            gold_lex = deriv.get_lexical()
            deriv_hat = Derivation.from_lexical(tags, gold_lex)
            evl.add(deriv_hat, deriv)
            with open(tab_out, 'a', newline='\n') as f:
                for gold_dln, pred_cat in zip(gold_lex, tags):
                    f.write(f'{pred_cat}\t{int(gold_dln.category1.equals(pred_cat))}\n')
            with open(args.out, 'a', newline='\n') as f:
                f.write(f'ID={ID} PARSER={args.tasks[0]} NUMPARSE=1\n')
                f.write(f'{deriv_hat}\n')
        ds.close()

    evl.eval_supertags()


if __name__ == '__main__':
    do_everything()
