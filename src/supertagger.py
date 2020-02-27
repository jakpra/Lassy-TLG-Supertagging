from Transformers.Transformer import Transformer
from Transformers.utils import FuzzyLoss, CustomLRScheduler, noam_scheme, make_mask as Mask

import torch
from torch import nn
from torch.nn import functional as F
from src.dataprep import TLGDataset
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from collections import Counter

import numpy as np

from src import dataprep

from typing import List, Any, Tuple, Union, Callable

FloatTensor = Union[torch.cuda.FloatTensor, torch.FloatTensor]
LongTensor = Union[torch.cuda.LongTensor, torch.LongTensor]


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
                 decoder_layers: int, d_intermediate: int, device: str, dropout: float=0.1, d_model: int = 300) -> None:
        super(Supertagger, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(num_classes=num_classes, encoder_heads=encoder_heads,
                                       decoder_heads=decoder_heads, encoder_layers=encoder_layers,
                                       decoder_layers=decoder_layers, d_model=d_model, d_intermediate=d_intermediate,
                                       dropout=dropout, device=device, reuse_embedding=True)
        self.device = device

    def forward(self, encoder_input: FloatTensor, decoder_input: FloatTensor, encoder_mask: LongTensor,
                decoder_mask: FloatTensor) -> FloatTensor:
        return self.transformer.forward(encoder_input, decoder_input, encoder_mask, decoder_mask)

    def train_epoch(self, dataset: TLGDataset, batch_size: int,
                    criterion: Callable[[FloatTensor, LongTensor], FloatTensor],
                    optimizer: optim.Optimizer, train_indices: List[int]) -> Tuple[float, int, int, int, int]:
        self.train()

        permutation = np.random.permutation(train_indices)

        batch_start = 0
        loss = 0.
        BS, BTS, BW, BTW = 0, 0, 0, 0

        # while batch_start < len(permutation):
        for i in range(len(permutation)):
            optimizer.zero_grad()
            # batch_end = min([batch_start + batch_size, len(permutation)])

            # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
            # batch_y = [dataset.Y[permutation[i]] for i in range(batch_start, batch_end)]
            batch_x = dataset.X[permutation[i]]
            batch_y = dataset.Y[permutation[i]]

            lens = list(map(len, batch_x))

            # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)
            # batch_y = pad_sequence(batch_y, batch_first=True).long().to(self.device)
            batch_e = F.embedding(batch_y.to(self.device), self.transformer.embedding_matrix)

            encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_x.shape[1])
            for i, l in enumerate(lens):
                encoder_mask[i, :, l::] = 0
            encoder_mask = encoder_mask.to(self.device)
            decoder_mask = Mask((batch_x.shape[0], batch_y.shape[1], batch_y.shape[1])).to(self.device)

            batch_p = self.forward(batch_x, batch_e, encoder_mask, decoder_mask)

            batch_loss = criterion(batch_p[:, :-1].permute(0, 2, 1), batch_y[:, 1:])
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

            (bs, bts), (bw, btw) = accuracy(argmaxes, y, dataset.type_dict['<PADDING>'])
            # (bs, bts), (bw, btw) = accuracy(batch_p[:, :-1].argmax(dim=-1), batch_y[:, 1:], dataset.type_dict['<PADDING>'])
            BS += bs
            BTS += bts
            BW += bw
            BTW += btw

            # batch_start += batch_size

        return loss, BS, BTS, BW, BTW

    def eval_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int], gen,
                   criterion: Callable[[FloatTensor, LongTensor], FloatTensor]) -> Tuple[float, int, int, int, int]:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0
            loss = 0.
            BS, BTS, BW, BTW = 0, 0, 0, 0
            n_words = 0

            gold_categories, generated_categories, correct_categories = Counter(), Counter(), Counter()

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

                encoder_mask = torch.ones(batch_y.shape[0], batch_y.shape[1], batch_x.shape[1])
                for i, l in enumerate(lens):
                    encoder_mask[i, :, l::] = 0
                encoder_mask = encoder_mask.to(self.device)
                batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict['<SEP>'])
                batch_loss = criterion(torch.log(batch_p[:, :-1]).permute(0, 2, 1), batch_y[:, 1:])
                loss += batch_loss.item()
                argmaxes = batch_p[:, :-1].argmax(dim=-1)
                y = batch_y[:, 1:]
                (bs, bts), (bw, btw) = accuracy(argmaxes, y, dataset.type_dict['<PADDING>'])
                BS += bs
                BTS += bts
                BW += bw
                BTW += btw

                categories_gold = gen.extract_outputs(y)
                categories_hat = gen.extract_outputs(argmaxes)
                for b, (sequence, sequence_gold) in enumerate(zip(categories_hat, categories_gold)):
                    # print(sequence, sequence_gold, file=sys.stderr)
                    for s, (cat, cat_gold) in enumerate(zip(sequence, sequence_gold)):
                        n_words += 1
                        # correct_index = [b, s] in correct_indices
                        # assert (cat == cat_gold) == (correct_index), (
                        #     b, s, cat, cat_gold, argmaxes[b, s], mask[b, s], y[b, s],
                        #     f'[{b}, {s}] {"not " if not correct_index else ""}in correct_indices')
                        if cat is None:
                            cat = 'None'
                        else:
                            msg = cat.validate()
                            if msg != 0:
                                if hasattr(gen, 'max_depth') and cat.depth() >= gen.max_depth:
                                    msg = 'Max depth reached'
                                    # print(b, s, msg, str(cat), cat.s_expr())
                                    cat = msg
                                elif hasattr(gen, 'max_len') and argmaxes.size(1) >= gen.max_len:
                                    msg = 'Max length reached'
                                    # print(b, s, msg, str(cat), cat.s_expr())
                                    cat = msg
                                else:
                                    # print(b, s, msg[0], str(cat), cat.s_expr(), file=sys.stderr)
                                    # print(argmaxes[b, s], file=sys.stderr)
                                    cat = msg[0]
                        gold_categories[str(cat_gold)] += 1
                        generated_categories[str(cat)] += 1
                        # if correct_index:
                        #     correct_categories[str(cat)] += 1

                # batch_start += batch_size

        return loss, BS, BTS, BW, BTW, gold_categories, generated_categories, n_words

    def infer_epoch(self, dataset: TLGDataset, batch_size: int, val_indices: List[int], max_len: int) \
            -> List[List[int]]:
        self.eval()

        with torch.no_grad():

            permutation = val_indices

            batch_start = 0

            P = []

            # while batch_start < len(permutation):
            for i in range(len(permutation)):
                # batch_end = min([batch_start + batch_size, len(permutation)])

                # batch_x = [dataset.X[permutation[i]] for i in range(batch_start, batch_end)]
                batch_x = dataset.X[permutation[i]]

                lens = list(map(len, batch_x))

                # batch_x = pad_sequence(batch_x, batch_first=True).to(self.device)

                encoder_mask = torch.ones(batch_x.shape[0], max_len * batch_x.shape[1], batch_x.shape[1])
                for i, l in enumerate(lens):
                    encoder_mask[i, :, l::] = 0
                encoder_mask = encoder_mask.to(self.device)
                batch_p = self.transformer.infer(batch_x, encoder_mask, dataset.type_dict['<SEP>'])
                batch_p = batch_p[:, :-1].argmax(dim=-1).cpu().numpy().tolist()
                P.append(batch_p)
                # batch_start += batch_size

        return P

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
                paths, scores = self.transformer.vectorized_beam_search(batch_x, encoder_mask, dataset.type_dict['<SEP>'],
                                                                        beam_width=beam_width)
                # todo: placeholder--take best beam as the only beam
                batch_p = paths[0]
                (bs, bts), (bw, btw) = accuracy(batch_p, batch_y[:, 1:], dataset.type_dict['<PADDING>'])
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
    from Transformers.utils import EncoderInput

    import ccg.util.argparse as ap

    args = ap.main()
    if args.cuda and torch.nn.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    # pretrained_path = 'type_LM'
    #
    # split_path = 'split.p'

    d_model = args.hidden_dims[0] if args.hidden_dims else 768
    dropout = args.dropout[0] if args.dropout else 0.2
    batch_size = args.batch_size
    epochs = args.epochs
    beam_size = 3

    # 3,4,4,128,0.1,0.1,4000
    if tlg is None:
        # tlg = DataPrep.do_everything_elmo(model_path
        #                     ='/home/kokos/Documents/Projects/Lassy/LassySupertagging/ELMoForManyLangs/Models/English',
        #                     data_file='data/XYZ_ccg.p')
        # tlg = dataprep.do_everything()
        tlg, train_indices, val_indices, test_indices, st = dataprep.do_everything_ccg(args, d_model)

    num_classes = len(tlg.type_dict) + 1
    print('Training on {} classes'.format(len(tlg.type_dict)))
    # n = Supertagger(num_classes, 4, 3, 3, 600, dropout=0.2, device='cuda', d_model=d_model)
    n = Supertagger(num_classes, encoder_heads=3, decoder_heads=8, encoder_layers=1,
                    decoder_layers=2, d_intermediate=d_model, device=args.device, dropout=dropout, d_model=d_model)

    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super(EncoderWrapper, self).__init__()
            self.encoder = enc

        def forward(self, x):
            return EncoderInput(encoder_input=self.encoder(vars(x.encoder_input), word_mask=x.mask, device=args.device)[0],
                                mask=x.mask)

    n.transformer.encoder = EncoderWrapper(st.span_encoder)

    if args.model_exists:
        with open(args.model, 'rb') as f:
            self_dict = n.state_dict()
            import re
            pretrained = torch.load(f)
            for k, p in pretrained.items():
                k = re.sub(r'network', 'transformer.decoder', k)
                k = re.sub(r'mha', 'mask_mha', k)
                k = re.sub(r'embedding_matrix', 'transformer.embedding_matrix', k)
                k = re.sub(r'predictor', 'transformer.predictor', k)
                if k in self_dict.keys():
                    self_dict[k] = p
                    print('replaced {}'.format(k))
                else:
                    continue
            n.load_state_dict(self_dict)
            del pretrained
    assert(all(list(map(lambda x: x.requires_grad, n.parameters()))))

    L = FuzzyLoss(torch.nn.KLDivLoss(reduction='batchmean'), num_classes, 0.1)

    a = optim.Adam(n.parameters(), betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-04)

    def var_rate(rate):
        return lambda _step, d_model, warmup_steps, batch_size=2048: \
            noam_scheme(_step=_step, d_model=d_model, warmup_steps=warmup_steps, batch_size=rate*batch_size)

    o = CustomLRScheduler(a, [noam_scheme], d_model=d_model, warmup_steps=4000, batch_size=4*batch_size)

    # with open(split_path, 'rb') as f:
    #     train_indices, val_indices, test_indices = pickle.load(f)

    best_val = 0.5

    for i in range(1000):
        loss, bs, bts, bw, btw = n.train_epoch(tlg, batch_size, L, o, train_indices)
        print('Epoch {}'.format(i))
        print(' Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(loss, bts/bs, btw/bw))
        if i % 5 == 0 and i != 0:
            loss, bs, bts, bw, btw, gold_categories, generated_categories, n_words = \
                n.eval_epoch(tlg, batch_size, val_indices, st.generators[0], L)
            print(' VALIDATION Loss: {}, Sentence Accuracy: {}, Word Accuracy: {}'.format(loss, bts / bs, btw / bw))

            print(f'most common gold categories (out of {n_words} in dev): '
                  f'{" | ".join(str(item) for item in gold_categories.most_common(10))}')
            print(f'most common generated categories (out of {n_words} in dev): '
                  f'{" | ".join(str(item) for item in generated_categories.most_common(10))}')
            # print(f'most common correct categories (out of {n_words} in dev): '
            #       f'{" | ".join(str(item) for item in correct_categories.most_common(10))}')

            bs, bts, bw, btw = n.eval_epoch_beam(tlg, batch_size, val_indices, beam_size)
            print(' BEAM VALIDATION Sentence Accuracy: {}, Word Accuracy: {}'.format(bts / bs, btw / bw))
            if bts/bs > best_val:
                best_val = bts/bs
                with open('model_{}_{}.p'.format(i, bts/bs), 'wb') as f:
                    torch.save(n.state_dict(), f)


if __name__ == '__main__':
    do_everything()
