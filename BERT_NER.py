import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score
from typing import List, Dict, Tuple


TRAIN_PATH  = "en_ewt-ud-train.iob2"
DEV_PATH    = "en_ewt-ud-dev.iob2"
TEST_PATH   = "en_ewt-ud-test-masked.iob2"
MODEL_NAME  = "bert-base-cased"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 3e-5
WARMUP_FRAC = 0.1


def parse_iob2(path: str) -> Tuple[List, List, List, List]:
    words_list, labels_list, comments_list, extras_list = [], [], [], []
    cur_words, cur_labels, cur_comments, cur_extras = [], [], [], []

    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                cur_comments.append(line)
                continue
            if not line.strip():
                if cur_words:
                    words_list.append(cur_words)
                    labels_list.append(cur_labels)
                    comments_list.append(cur_comments)
                    extras_list.append(cur_extras)
                    cur_words, cur_labels, cur_comments, cur_extras = [], [], [], []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                continue
            cur_words.append(parts[1])
            cur_labels.append(parts[2])
            cur_extras.append("\t".join(parts[3:]) if len(parts) > 3 else "-\t-")

    if cur_words:
        words_list.append(cur_words)
        labels_list.append(cur_labels)
        comments_list.append(cur_comments)
        extras_list.append(cur_extras)

    return words_list, labels_list, comments_list, extras_list


def build_label_vocab(labels_list: List[List[str]]) -> Tuple[Dict, Dict]:
    unique = sorted({lbl for sent in labels_list for lbl in sent if lbl != "O"})
    label_list = ["O"] + unique
    print("Label list: ", label_list)
    label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
    id2label  = {idx: lbl for lbl, idx in label2id.items()}
    return label2id, id2label


def write_iob2(path: str, words_list: List, pred_labels_list: List,
               comments_list: List, extras_list: List) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for i, (words, labels) in enumerate(zip(words_list, pred_labels_list)):
            for comment in comments_list[i]:
                fh.write(comment + "\n")
            for j, (word, label) in enumerate(zip(words, labels)):
                extra = "\t" + extras_list[i][j] if j < len(extras_list[i]) else "\t-\t-"
                fh.write(f"{j + 1}\t{word}\t{label}{extra}\n")
            fh.write("\n")
    print(f"Predictions written to {path} ({len(words_list)} sentences)")


class NERDataset(Dataset):
    def __init__(self, words_list, labels_list, tokenizer, label2id):
        self.label2id = label2id
        self.encodings = tokenizer(
            words_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        self.labels = self._align_labels(labels_list)

    def _align_labels(self, labels_list):
        aligned = []
        for i, labels in enumerate(labels_list):
            word_ids = self.encodings.word_ids(batch_index=i)
            label_ids, prev = [], None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev:
                    label_ids.append(self.label2id[labels[word_id]])
                else:
                    label_ids.append(-100)
                prev = word_id
            aligned.append(label_ids)
        return aligned

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()
                if k != "offset_mapping"}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class TestDataset(Dataset):
    def __init__(self, words_list, tokenizer):
        self.words_list = words_list
        self.encodings = tokenizer(
            words_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )

    def __len__(self):
        return len(self.words_list)

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()
                if k != "offset_mapping"}


class BertForNER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        logits = self.classifier(self.dropout(out.last_hidden_state))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.shape[-1]), labels.view(-1)
            )
        return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_words, train_labels, _, _ = parse_iob2(TRAIN_PATH)
    label2id, id2label = build_label_vocab(train_labels)
    print(f"Loaded {len(train_words)} training sentences, {len(label2id)} labels")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        NERDataset(train_words, train_labels, tokenizer, label2id),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    dev_words, dev_labels, _, _ = parse_iob2(DEV_PATH)
    print(f"Loaded {len(dev_words)} dev sentences")
    dev_loader = DataLoader(
        NERDataset(dev_words, dev_labels, tokenizer, label2id),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = BertForNER(num_labels=len(label2id)).to(device)
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": LR},
        {"params": list(model.classifier.parameters()) +
                   list(model.dropout.parameters()), "lr": LR * 10},
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * EPOCHS * WARMUP_FRAC),
        num_training_steps=len(train_loader) * EPOCHS,
    )

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids",
                               torch.zeros_like(batch["input_ids"])).to(device),
                labels=batch["labels"].to(device),
            )
            output["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += output["loss"].item()
            if step % 100 == 0 or step == num_batches:
                print(f"  Epoch {epoch}/{EPOCHS}  batch {step}/{num_batches}"
                      f"  loss={total_loss / step:.4f}")

        model.eval()
        val_loss = 0.0
        all_preds, all_golds = [], []
        with torch.no_grad():
            for batch in dev_loader:
                output = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch.get("token_type_ids",
                                             torch.zeros_like(batch["input_ids"])).to(device),
                    labels=batch["labels"].to(device),
                )
                val_loss += output["loss"].item()
                preds = torch.argmax(output["logits"], dim=-1)
                for pred_seq, label_seq in zip(preds.cpu().numpy(),
                                               batch["labels"].numpy()):
                    pred_tags, gold_tags = [], []
                    for p, g in zip(pred_seq, label_seq):
                        if g == -100:
                            continue
                        pred_tags.append(id2label[p])
                        gold_tags.append(id2label[g])
                    all_preds.append(pred_tags)
                    all_golds.append(gold_tags)
        val_loss /= len(dev_loader)
        val_f1 = f1_score(all_golds, all_preds)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            torch.save({"model_state_dict": model.state_dict(),
                        "label2id": label2id, "id2label": id2label},
                        "bert_ner_best.pt")
        print(f"Epoch {epoch}/{EPOCHS} complete  "
              f"train_loss={total_loss / num_batches:.4f}  "
              f"val_loss={val_loss:.4f}  f1={val_f1:.4f}"
              f"  {'(best — saved)' if improved else '(no improvement)'}\n")

    print("Loading best checkpoint for inference ...")
    checkpoint = torch.load("bert_ner_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_words, _, test_comments, test_extras = parse_iob2(TEST_PATH)
    print(f"Loaded {len(test_words)} test sentences, running inference ...")

    test_dataset = TestDataset(test_words, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []
    num_batches = len(test_loader)
    with torch.no_grad():
        for step, batch in enumerate(test_loader, 1):
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch.get("token_type_ids",
                               torch.zeros_like(batch["input_ids"])).to(device),
            )["logits"]
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            if step % 100 == 0 or step == num_batches:
                print(f"  Inference batch {step}/{num_batches}")

    pred_labels_list = []
    for i, words in enumerate(test_words):
        word_ids = test_dataset.encodings.word_ids(batch_index=i)
        seen, sent_labels = set(), []
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen.add(word_id)
            sent_labels.append(id2label[all_preds[i][token_idx]])
        pred_labels_list.append(sent_labels)

    write_iob2("predictions.iob2", test_words, pred_labels_list,
               test_comments, test_extras)


if __name__ == "__main__":
    main()