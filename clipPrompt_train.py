import h5py
from trainers.options import params
from trainers.clipPrompt.promptclip import FewshotCLIP

if params.domain == 'single':
    from trainers.dataloader import *
else:
    raise AssertionError

# ---------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------

# Generate sentence from tokens
h5loader_train = h5py.File(params.root + params.dataset_name + '/train.h5', 'r')
h5loader_valid = h5py.File(params.root + params.dataset_name + '/test.h5', 'r')
if params.dataset_name == 'shoes':
    ImageData = ShoesData
elif params.dataset_name == 'fashionIQ':
    ImageData = FashionIQData
elif params.dataset_name == 'fashion200k':
    ImageData = Fashion200kData
elif params.dataset_name == 'MIT':
    ImageData = MITData
elif params.dataset_name == 'celebA':
    ImageData = CelebAData
elif params.dataset_name == 'B2W':
    ImageData = B2WData
else:
    raise AssertionError

# Seed rng for reproducibility
random.seed(params.seed)
torch.manual_seed(params.seed)
if params.cuda:
    torch.cuda.manual_seed_all(params.seed)

# Setup dataloader
splits = ['train', 'valid', 'test']
train_dataset = ImageData(params, 'train', h5loader_train)
train_loader = DataLoader(train_dataset, collate_fn=cap_collate_fn,
                          batch_size=params.batch_size, shuffle=True, num_workers=params.n_works)
valid_dataset = ImageData(params, 'test', h5loader_valid)
valid_loader = DataLoader(valid_dataset, collate_fn=cap_collate_fn,
                          batch_size=128, shuffle=False, num_workers=params.n_works)
print('train sample number:', len(train_dataset))
print('valid sample number:', len(valid_dataset))

# Setup model
model = FewshotCLIP(params)
opt = torch.optim.Adam(model.parameters(), lr=params.lr)
if params.cuda:
    model = model.cuda()
print(model)
assert 0 == 1

# Setup global variable
iter_ = 0
best_score = 0

# Logging
fprefix = f'PromptCLIP/{params.seed}/{params.dataset_name}/nshot_{params.n_shot}+nref_{params.n_ref}' \
          f'+beta_{params.beta}+lctx_{params.l_ctx}+rctx_{params.r_ctx}+bs_{params.batch_size}+pos_{params.pos}/'
os.makedirs(fprefix, exist_ok=True)
with open(fprefix + 'valid_curve.csv', 'w') as f:
    f.write('epoch,idx,R1,R5,R10,R20,R50,R100\n')


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def batch_data(entry):
    ref_features = entry['ref_features'].float()
    tgt_features = entry['tgt_features'].float()
    labels = entry['labels']
    compare = entry['compare']
    tgt_text = entry['tgt_text']
    if params.cuda:
        ref_features = ref_features.cuda(non_blocking=True)
        tgt_features = tgt_features.cuda(non_blocking=True)
    return ref_features, tgt_features, labels, compare, tgt_text


for epoch in range(params.epochs):
    for idx, batch in enumerate(train_loader):
        # Moving current batch to GPU, if available
        ref_features, tgt_features, labels, compare, tgt_text = batch_data(batch)

        logits, ref_logits, tgt_logits, _, _ = model(ref_features, tgt_features, labels, compare, tgt_text)
        loss = model.contrastive_loss(logits).mean() * (1 - params.beta) + \
               model.contrastive_loss(ref_logits).mean() * params.beta * .5 + \
               model.contrastive_loss(tgt_logits).mean() * params.beta * .5

        opt.zero_grad()
        loss.backward()
        opt.step()

        print('epoch', epoch, 'idx', idx, 'iter', iter_, 'loss', loss.item())

        if iter_ % params.freq == 0 and iter_ > 0:
            with torch.no_grad():
                model.eval()
                score = []
                predict_features = []
                target_features = []
                for idx_val, batch in enumerate(valid_loader):
                    if idx_val * params.batch_size == params.test_size:
                        break
                    ref_features, tgt_features, labels, compare, tgt_text = batch_data(batch)
                    _, _, _, pred_feat, tgt_feat = \
                        model(ref_features, tgt_features, labels, compare, tgt_text, train=False)
                    predict_features.append(pred_feat.cpu())
                    target_features.append(tgt_feat.cpu())
                predict_features = torch.cat(predict_features, dim=0)
                target_features = torch.cat(target_features, dim=0)
                scores = predict_features @ target_features.t()
                _, ranks = torch.sort(scores, dim=1, descending=True)
                sample_indexes = torch.arange(len(scores)).view(-1, 1)
                ranks = (ranks == sample_indexes).nonzero()[:, 1]
                R1 = (ranks < 1).float().mean()
                R5 = (ranks < 5).float().mean()
                R10 = (ranks < 10).float().mean()
                R20 = (ranks < 20).float().mean()
                R50 = (ranks < 50).float().mean()
                R100 = (ranks < 100).float().mean()

                output_score = R50

                print('epoch', epoch, 'idx_val', idx_val, 'iter', iter_)
                print('R1:', R1, 'R5:', R5, 'R10:', R10, 'R20:', R20, 'R50:', R50, 'R100:', R100)

                with open(fprefix + 'valid_curve.csv', 'a') as f:
                    f.write(
                        f'{epoch},{idx},{R1.item()},{R5.item()},{R10.item()},{R20.item()},{R50.item()},{R100.item()}\n')
                if best_score < output_score:
                    best_score = output_score.item()
                    torch.save(model.prompt_model.state_dict(), fprefix + 'model.pt')
                model.train()

        iter_ += 1
