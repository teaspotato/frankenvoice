"""Fine-tune a Moonshine encoder to name the first word of an overlap from a fixed window.

Input: mic audio from 0.5 s before the VAD onset to W after it (W = 0.25-1.0 s in
training, so one model serves any decision time). Outputs: the first word's class
(vocab.CLASSES) and p(more), whether speech goes on past that word inside the window.

Training mixes synthetic clips (synth.py) under real agent echo and mic noise cut from
recorded sessions, plus the real word onsets from those sessions (cut_real.py).
Evaluates on the 10 labeled overlaps (eval_set.py) at several W.

  python train.py --synth synth.npz [rin_takes.npz] --real real_train.npz --eval eval_bugs2.npz --out ckpt/
  python train.py ... --export model.onnx   # encoder + head, dynamic length, for onnxruntime-web
"""
import argparse, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from vocab import BACKCHANNEL, CLASSES, FLOOR_OPENER, IDX

HOLDOUT = ['af_nova', 'bm_george', 'jf_alpha']  # synth.HOLDOUT: voices kept out of training

SR = 16000
PRE = 0.5  # audio before the anchor (VAD onset); speech often starts 0.1-0.4 s before Silero fires


class Spotter(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import MoonshineModel
        self.enc = MoonshineModel.from_pretrained(name).get_encoder()
        h = self.enc.config.hidden_size
        self.att = nn.Linear(h, 1)
        self.cls = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, len(CLASSES)))
        self.more = nn.Sequential(nn.Linear(h, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x, mask=None):
        # x: (B, T) 16 kHz; mask: (B, T) 1 for real samples. Moonshine normalizes internally.
        hs = self.enc(x, attention_mask=mask).last_hidden_state if mask is not None else self.enc(x).last_hidden_state
        a = self.att(hs).squeeze(-1)
        if mask is not None:
            fm = F.interpolate(mask[:, None].float(), size=hs.shape[1], mode='nearest')[:, 0]
            a = a.masked_fill(fm < 0.5, -1e4)
        w = a.softmax(-1)[..., None]
        z = (w * hs).sum(1)
        return self.cls(z), self.more(z).squeeze(-1)


def db(x):
    return 10 ** (x / 20)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-9)


class Data:
    def __init__(self, synth, real, seed=0, holdout=False, holdout_voices=HOLDOUT):
        self.rng = np.random.default_rng(seed)
        self.s = {k: [] for k in ('audio', 'len', 'y', 'first_end', 'more')}
        for f in synth:  # synth.py output and split_takes.py output share one format
            s = np.load(f)
            keep = np.isin(s['voice'], holdout_voices) == holdout
            for k in self.s:
                v = s[k][keep]
                self.s[k] += list(v) if k == 'audio' else [v]
        self.s = {k: (v if k == 'audio' else np.concatenate(v)) for k, v in self.s.items()}
        r = np.load(real)
        self.rpre = float(r['pre'])
        self.r = {k: r[k] for k in ('audio', 'agent', 'y', 'dur', 'next')}
        none = r['y'] == IDX['none']
        self.noise = r['audio'][none]  # mic with no user speech: room noise + echo residue
        self.agent = r['agent'][r['overlap']]  # agent playback on the mic clock

    def background(self, n):
        rng = self.rng
        bg = np.zeros(n, np.float32)
        nz = self.noise[rng.integers(len(self.noise))].astype(np.float32)
        i = rng.integers(0, max(1, len(nz) - n)); seg = nz[i:i + n]
        bg[:len(seg)] += seg * db(rng.uniform(-6, 6))
        if rng.random() < 0.8:  # echo residue the browser AEC lets through
            ag = self.agent[rng.integers(len(self.agent))].astype(np.float32)
            i = rng.integers(0, max(1, len(ag) - n)); seg = ag[i:i + n]
            k = rng.integers(1, 6)
            seg = np.convolve(seg, np.ones(k) / k, 'same')  # random low-pass
            bg[:len(seg)] += seg * db(rng.uniform(-40, -15))
        return bg

    def synth_example(self, W):
        rng = self.rng
        i = rng.integers(len(self.s['y']))
        x = self.s['audio'][i][:self.s['len'][i]].astype(np.float32)
        lag = rng.uniform(0.0, 0.45)  # VAD onset comes after the speech starts
        n = int((PRE + W) * SR)
        out = self.background(n)
        st = int((PRE - lag) * SR)
        seg = x[:max(0, n - st)]
        g = db(rng.uniform(-38, -14)) / rms(x)
        out[st:st + len(seg)] += seg * g
        fe = self.s['first_end'][i] - lag  # first word end, relative to the anchor
        more = bool(self.s['more'][i]) and fe + 0.12 < W
        return out, int(self.s['y'][i]), more

    def real_example(self, W):
        rng = self.rng
        i = rng.integers(len(self.r['y']))
        x = self.r['audio'][i].astype(np.float32); y = int(self.r['y'][i])
        lag = rng.uniform(0.0, 0.45) if y != IDX['none'] else rng.uniform(0, 0.5)
        a = int((self.rpre + lag - PRE) * SR); n = int((PRE + W) * SR)
        if a < 0:  # the window starts before the cut does: pad with silence; background() adds noise below
            x = np.concatenate([np.zeros(-a, np.float32), x]); a = 0
        out = x[a:a + n]
        out = np.pad(out, (0, n - len(out)))
        more = y != IDX['none'] and self.r['next'][i] - lag + 0.05 < W
        out = out * db(rng.uniform(-6, 6)) + 0.3 * self.background(n)
        return out, y, bool(more)

    def augment(self, x):
        rng = self.rng
        if rng.random() < 0.3:  # browser noise suppression gating parts of the speech
            for _ in range(rng.integers(1, 3)):
                a = rng.integers(0, len(x)); b = a + rng.integers(800, 3200)
                x[a:b] *= db(rng.uniform(-40, -15))
        return x

    def batch(self, B, p_real=0.25):
        W = float(self.rng.uniform(0.25, 1.0))
        xs, ys, ms = [], [], []
        for _ in range(B):
            r = self.rng.random()
            if r < 0.06:  # pure background
                x, y, m = self.background(int((PRE + W) * SR)), IDX['none'], False
            elif r < 0.06 + p_real:
                x, y, m = self.real_example(W)
            else:
                x, y, m = self.synth_example(W)
            xs.append(self.augment(x)); ys.append(y); ms.append(m)
        return torch.tensor(np.stack(xs)), torch.tensor(ys), torch.tensor(ms, dtype=torch.float32)


def decide(c, pm, th=0.5):
    """Map (first word class, p(more)) to what the agent does now. 'wait' = keep deciding."""
    if c in FLOOR_OPENER:
        return 'floor'
    if c in BACKCHANNEL and pm < th:
        return 'backchannel'
    if c == 'none':
        return 'backchannel'  # noise: resume
    return 'wait'


@torch.no_grad()
def val_synth(model, data, dev, n=8):
    """First-word accuracy on held-out voices, mixed like training (fixed seed)."""
    model.eval(); data.rng = np.random.default_rng(123); hit = tot = 0
    for _ in range(n):
        x, y, m = data.batch(32, p_real=0)
        hit += (model(x.to(dev))[0].argmax(-1).cpu() == y).sum().item(); tot += len(y)
    model.train()
    return hit / tot


@torch.no_grad()
def evaluate(model, path, dev, Ws=(0.3, 0.4, 0.5, 0.6, 0.8, 1.0)):
    d = np.load(path); pre = float(d['pre'])
    model.eval(); res = {}
    for W in Ws:
        x = d['audio'][:, int((pre - PRE) * SR):int((pre + W) * SR)].astype(np.float32)
        lg, mo = model(torch.tensor(x, device=dev))
        p = lg.softmax(-1).cpu().numpy(); pm = torch.sigmoid(mo).cpu().numpy()
        rows = []
        for k in range(len(x)):
            c = CLASSES[int(p[k].argmax())]
            rows.append(dict(text=str(d['text'][k]), truth=CLASSES[int(d['y'][k])], pred=c, p=float(p[k].max()), more=float(pm[k]),
                             decision=str(d['decision'][k]), act=decide(c, pm[k])))
        acc = np.mean([r['pred'] == r['truth'] for r in rows])
        wrong = sum(1 for r in rows if (r['act'] == 'backchannel' and r['decision'] != 'backchannel') or
                    (r['act'] == 'floor' and r['decision'] == 'backchannel'))
        res[W] = dict(first_word_acc=float(acc), wrong_actions=wrong, decided=sum(r['act'] != 'wait' for r in rows), rows=rows)
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synth', required=True, nargs='+'); ap.add_argument('--real', required=True); ap.add_argument('--eval', required=True)
    ap.add_argument('--model', default='UsefulSensors/moonshine-tiny')
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--freeze-encoder', action='store_true')
    ap.add_argument('--p-real', type=float, default=0.25)
    ap.add_argument('--holdout', help='comma-separated voices (or a prefix ending in *) kept out of training and scored')
    ap.add_argument('--out', default='ckpt'); ap.add_argument('--export')
    ap.add_argument('--load', help='skip training, load this state dict')
    ap.add_argument('--test-synth', nargs='+', help='synthetic sets scored in full (every voice unseen), e.g. another TTS engine')
    ap.add_argument('--test-voices', help='with --test-synth: only these voices (comma list or prefix*)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    model = Spotter(a.model).to(dev)
    if a.load:
        model.load_state_dict(torch.load(a.load, map_location=dev))
    else:
        hv = HOLDOUT
        if a.holdout:
            allv = sorted({str(v) for f in a.synth for v in np.load(f)['voice']})
            hv = [v for v in allv if any(v == h or (h.endswith('*') and v.startswith(h[:-1])) for h in a.holdout.split(','))]
            print('held-out voices:', hv)
        data = Data(a.synth, a.real, holdout_voices=hv)
        val = Data(a.synth, a.real, holdout=True, holdout_voices=hv)
        enc_p = list(model.enc.parameters()); head_p = [p for n, p in model.named_parameters() if not n.startswith('enc.')]
        if a.freeze_encoder:
            for p in enc_p: p.requires_grad = False
        opt = torch.optim.AdamW([{'params': enc_p, 'lr': a.lr}, {'params': head_p, 'lr': a.lr * 20}], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))
        log = []; t0 = time.time()
        for step in range(1, a.steps + 1):
            x, y, m = data.batch(a.batch, a.p_real)
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            lg, mo = model(x)
            loss = F.cross_entropy(lg, y, label_smoothing=0.05) + 0.5 * F.binary_cross_entropy_with_logits(mo, m)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            if step % 100 == 0:
                acc = (lg.argmax(-1) == y).float().mean().item()
                print(f'step {step} loss {loss.item():.3f} batch acc {acc:.2f} {time.time() - t0:.0f}s', flush=True)
            if step % 500 == 0 or step == a.steps:
                ev = evaluate(model, a.eval, dev)
                summ = {W: (r['first_word_acc'], r['decided'], r['wrong_actions']) for W, r in ev.items()}
                vs = val_synth(model, val, dev)
                print(f'  held-out voices first-word acc {vs:.2f}; Bugs 2 (first-word acc, decided, wrong actions) by W:', summ, flush=True)
                log.append(dict(step=step, val_synth=vs, eval=summ))
        torch.save(model.state_dict(), f'{a.out}/spotter.pt')
        json.dump(log, open(f'{a.out}/log.json', 'w'))
    if a.test_synth:
        for f in a.test_synth:
            allv = sorted({str(v) for v in np.load(f)['voice']})
            pick = allv if not a.test_voices else [v for v in allv if any(
                v == h or (h.endswith('*') and v.startswith(h[:-1])) for h in a.test_voices.split(','))]
            print(f'{f}: first-word acc {val_synth(model, Data([f], a.real, holdout=True, holdout_voices=pick), dev, n=30):.3f} '
                  f'on {len(pick)} voices', flush=True)
    ev = evaluate(model, a.eval, dev)
    json.dump(ev, open(f'{a.out}/eval.json', 'w'), indent=1)
    for W, r in ev.items():
        print(f'W={W}: first word {r["first_word_acc"]:.0%}, decided {r["decided"]}/10, wrong actions {r["wrong_actions"]}')
        for row in r['rows']:
            print(f'   {row["text"][:30]:30s} truth {row["truth"]:7s} pred {row["pred"]:7s} p={row["p"]:.2f} more={row["more"]:.2f} -> {row["act"]} ({row["decision"]})')
    if a.export:
        model.eval().cpu()
        x = torch.zeros(1, int((PRE + 0.5) * SR))
        torch.onnx.export(model, (x,), a.export, input_names=['audio'], output_names=['logits', 'more'],
                          dynamic_axes={'audio': {0: 'b', 1: 't'}, 'logits': {0: 'b'}, 'more': {0: 'b'}}, opset_version=18, dynamo=True)
        json.dump({'classes': CLASSES, 'pre_s': PRE, 'sample_rate': SR}, open(a.export + '.json', 'w'))
        print('exported', a.export)


if __name__ == '__main__':
    main()
