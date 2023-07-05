import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.dataloader import CMP_NAMES, CLASS_NAMES

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg):
        super(PromptLearner, self).__init__()
        self.ref_ln = nn.Sequential(
            nn.Linear(512, 512 * cfg.n_ref),
            nn.Dropout(0.1)
        )
        num = 1
        if cfg.csc:
            ctx_vectors = torch.empty(32, cfg.l_ctx * num, 512)
            edt_vectors = torch.empty(32, cfg.r_ctx * num, 512)
        else:
            ctx_vectors = torch.empty(cfg.l_ctx * num, 512)
            edt_vectors = torch.empty(cfg.r_ctx * num, 512)
        nn.init.normal_(ctx_vectors, std=0.02)
        nn.init.normal_(edt_vectors, std=0.02)
        self.b = torch.nn.Parameter(torch.ones(1, 2, 512, dtype=torch.float))
        self.ctx_prompt = nn.Parameter(ctx_vectors)
        self.edt_prompt = nn.Parameter(edt_vectors)
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 0.0]))

    def forward(self, comp_features, edit_features, img_features):
        f = torch.sigmoid(edit_features) * img_features * self.a[0] + comp_features * self.a[1]
        return f


class FewshotCLIP(nn.Module):
    def __init__(self, cfg):
        super(FewshotCLIP, self).__init__()
        self.cfg = cfg
        self.classnames = CLASS_NAMES[cfg.dataset_name]

        print(f"Loading CLIP (backbone: {cfg.backbone})")
        clip_model = torch.jit.load('clip/backbones/' + cfg.model_path + '.pt', "cpu").eval()
        clip_model = clip.build_model(clip_model.state_dict())
        self.text_encoder = TextEncoder(clip_model)
        self.clip_model = clip_model
        self.prompt_model = PromptLearner(cfg)
        if cfg.prompt_path:
            self.prompt_model.load_state_dict(torch.load(cfg.prompt_path))

        if cfg.cuda:
            self.clip_model = self.clip_model.cuda()
            self.text_encoder = self.text_encoder.cuda()
            self.prompt_model = self.prompt_model.cuda()

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad_(False)
        for name, param in self.clip_model.named_parameters():
            param.requires_grad_(False)

    def prompt_learner(self, cls, cmp, tgt_text, ref_prompt, comp=True):
        # select mode
        n_ctx = self.cfg.l_ctx if comp else self.cfg.r_ctx
        prompt_model = self.prompt_model.ctx_prompt if comp else self.prompt_model.edt_prompt

        # construct template
        dtype = self.clip_model.dtype
        compares = [CMP_NAMES[i] for i in cmp]
        classnames = [self.classnames[i] for i in cls]
        categories = [z[0] for z in tgt_text]
        ref_placeholder = ' '.join(['#'] * self.cfg.n_ref)
        ctx_placeholder = ''
        attributes = [[x] + [y] + z[2:] for x, y, z in zip(compares, classnames, tgt_text)]
        prompts = []

        # Prompting
        if self.cfg.dataset_name in ['fashionIQ', 'B2W', 'celebA']:
            for texts in attributes:
                prefix, append = ['reference', ref_placeholder], []
                attr = [str(t) for t in texts]
                if n_ctx > 0:
                    if self.cfg.pos == 'front':
                        prefix.append(ctx_placeholder)
                    elif self.cfg.pos == 'end':
                        append.append(ctx_placeholder)
                    elif self.cfg.pos == 'middle':
                        attr = attr[:2] + [ctx_placeholder] + attr[2:]
                    else:
                        raise NotImplementedError
                prompts.append(' '.join(prefix + attr + append))
        else:
            raise NotImplementedError

        # tokenize template
        with torch.no_grad():
            tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
            if self.cfg.cuda:
                tokenized_prompts = tokenized_prompts.cuda()
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        # replace placeholder with embeddings
        prompts = []
        for i, (cat, token, emb) in enumerate(zip(categories, tokenized_prompts, embedding)):
            prompt = []
            count_x, count_y = 0, 0
            for t, e in zip(token, emb):
                if t == 265:
                    prompt.append(prompt_model[count_x])
                    count_x += 1
                elif t == 258:
                    prompt.append(ref_prompt[i, count_y])
                    count_y += 1
                else:
                    prompt.append(e)
            prompts.append(torch.stack(prompt))
        prompts = torch.stack(prompts)

        return self.text_encoder(prompts, tokenized_prompts)

    def prompt_sidelearner(self, cmp, ref_prompt):
        dtype = self.clip_model.dtype
        compares = [CMP_NAMES[i] for i in cmp]
        ref_placeholder = ' '.join(['#'] * self.cfg.n_ref)
        prompts = [' '.join(['reference', ref_placeholder]) for _ in compares]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        if self.cfg.cuda:
            tokenized_prompts = tokenized_prompts.cuda()

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        prompts = []
        for i, (token, emb) in enumerate(zip(tokenized_prompts, embedding)):
            prompt = []
            count_x, count_y = 0, 0
            for t, e in zip(token, emb):
                if t == 258:
                    prompt.append(ref_prompt[i, count_y])
                    count_y += 1
                else:
                    prompt.append(e)
            prompts.append(torch.stack(prompt))
        prompts = torch.stack(prompts)

        return self.text_encoder(prompts, tokenized_prompts)

    def forward(self, ref, tgt, cls, cmp, tgt_text, train=True):
        ref_features = self.clip_model.encode_image(ref)
        tgt_features = self.clip_model.encode_image(tgt)
        ref_prompt = self.prompt_model.ref_ln(ref_features).view(-1, self.cfg.n_ref, 512)

        comp_features = self.prompt_learner(cls, cmp, tgt_text, ref_prompt.data, comp=True)
        edit_features = self.prompt_learner(cls, cmp, tgt_text, ref_prompt.data, comp=False)

        text_features = self.prompt_model(comp_features, edit_features, ref_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if train:
            rtxt_features = self.prompt_sidelearner(cmp, ref_prompt)
            rtxt_features = rtxt_features / rtxt_features.norm(dim=-1, keepdim=True)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)

            tgt_prompt = self.prompt_model.ref_ln(tgt_features).view(-1, self.cfg.n_ref, 512)
            ttxt_features = self.prompt_sidelearner(cmp, tgt_prompt)
            ttxt_features = ttxt_features / ttxt_features.norm(dim=-1, keepdim=True)
            tgt_features = tgt_features / tgt_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = logit_scale * tgt_features @ text_features.t()
            ref_logits = logit_scale * ref_features @ rtxt_features.t()
            tgt_logits = logit_scale * tgt_features @ ttxt_features.t()
        else:
            tgt_features = tgt_features / tgt_features.norm(dim=-1, keepdim=True)
            logits, ref_logits, tgt_logits = None, None, None

        return logits, ref_logits, tgt_logits, text_features, tgt_features

    def contrastive_loss(self, scores):
        # scores: (bsize, bsize)
        bsize, _ = scores.size()
        diagonal = scores.diag().view(bsize, 1)
        cost_s = (self.cfg.margin + scores - diagonal).clamp(min=0)
        cost_im = (self.cfg.margin + scores - diagonal.T).clamp(min=0)

        # clear diagonals
        cost_s = cost_s.fill_diagonal_(0)
        cost_im = cost_im.fill_diagonal_(0)

        # keep the maximum violating negative for each query
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

        return cost_s + cost_im
