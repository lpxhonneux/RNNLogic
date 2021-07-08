import torch
from torch.utils.data import Dataset
from torch_sparse import spmm
import rnnlogic_ext as pyrnnlogic
import numpy as np
import os
import gc
import random

class KnowledgeGraph:
    def __init__(self, args):
        self.args = args
        data_path = args.data_path

        self.entity2id = dict()
        self.relation2id = dict()

        with open(os.path.join(data_path, 'entities.dict')) as fi:
            for line in fi:
                id, entity = line.strip().split('\t')
                self.entity2id[entity] = int(id)

        with open(os.path.join(data_path, 'relations.dict')) as fi:
            for line in fi:
                id, relation = line.strip().split('\t')
                self.relation2id[relation] = int(id)

        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        
        self.train_facts = list()
        self.valid_facts = list()
        self.test_facts = list()
        self.hr2o = dict()
        self.hr2v = dict()
        self.relation2sparse = [[[], []] for k in range(self.relation_size)]
        self.relation2hr2i = [dict() for k in range(self.relation_size)]

        with open(os.path.join(data_path, "train.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.train_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)
                if hr_index not in self.hr2o:
                    self.hr2o[hr_index] = list()
                self.hr2o[hr_index].append(t)
                if hr_index not in self.hr2v:
                    self.hr2v[hr_index] = list()
                self.hr2v[hr_index].append(t)

                self.relation2sparse[r][0].append(t)
                self.relation2sparse[r][1].append(h)

                i = len(self.relation2hr2i[r])
                ht_index = self.encode_ht(h, t)
                self.relation2hr2i[r][ht_index] = i

        with open(os.path.join(data_path, "valid.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.valid_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)
                if hr_index not in self.hr2v:
                    self.hr2v[hr_index] = list()
                self.hr2v[hr_index].append(t)

        with open(os.path.join(data_path, "test.txt")) as fi:
            for line in fi:
                h, r, t = line.strip().split('\t')
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                self.test_facts.append((h, r, t))

                hr_index = self.encode_hr(h, r)
                if hr_index not in self.hr2v:
                    self.hr2v[hr_index] = list()
                self.hr2v[hr_index].append(t)

        for r in range(self.relation_size):
            index = torch.LongTensor(self.relation2sparse[r]).cuda()
            value = torch.ones(index.size(1)).cuda()
            self.relation2sparse[r] = [index, value]

        print("Data loading | DONE!")

    def encode_hr(self, h, r):
        return r * self.entity_size + h

    def decode_hr(self, index):
        h, r = index % self.entity_size, index // self.entity_size
        return h, r

    def encode_ht(self, h, t):
        return t * self.entity_size + h

    def decode_hr(self, index):
        h, t = index % self.entity_size, index // self.entity_size
        return h, t

    def get_updated_adjacency(self, r, edges_to_remove):
        if edges_to_remove == None:
            return None
        index = self.relation2sparse[r][0]
        value = self.relation2sparse[r][1]
        mask = (index.unsqueeze(1) == edges_to_remove.unsqueeze(-1))
        mask = mask.all(dim=0).any(dim=0)
        mask = ~mask
        index = index[:, mask]
        value = value[mask]
        return [index, value]

    def grounding(self, h, r, rule, updated_adjacency):
        with torch.no_grad():
            #x = torch.LongTensor(h).cuda()
            #x = torch.nn.functional.one_hot(x, self.entity_size).transpose(0, 1)
            x = torch.nn.functional.one_hot(h, self.entity_size).transpose(0, 1)
            for r_body in rule:
                index = self.relation2sparse[r_body][0]
                value = self.relation2sparse[r_body][1]
                if r_body == r and updated_adjacency != None:
                    index = updated_adjacency[0]
                    value = updated_adjacency[1]
                x = spmm(index, value, self.entity_size, self.entity_size, x)
        return x

class TrainDataset(Dataset):
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg

        self.r2queries = [[] for r in range(self.kg.relation_size)]
        for h, r, t in self.kg.train_facts:
            self.r2queries[r].append((h, r))

        self.r2queries = [list(set(_)) for _ in self.r2queries]

        self.make_batches()

    def make_batches(self):
        for r in range(self.kg.relation_size):
            random.shuffle(self.r2queries[r])

        self.batches = list()
        for r, queries in enumerate(self.r2queries):
            for k in range(0, len(queries), self.args.predictor_batch_size):
                start = k
                end = min(k + self.args.predictor_batch_size, len(queries))
                self.batches.append(queries[start:end])
        random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def get_batch(self, idx):
        #idx = random.randint(0, len(self.batches) - 1)
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        target = torch.zeros(len(data), self.kg.entity_size)
        edges = list()
        for k, (h, r) in enumerate(data):
            hr_index = self.kg.encode_hr(h, r)
            t_index = torch.LongTensor(self.kg.hr2o[hr_index])
            target[k][t_index] = 1
            h_index = torch.full_like(t_index, h)
            edges_ = torch.stack([t_index, h_index], dim=0)
            edges_.size()
            edges.append(edges_)
        edges = torch.cat(edges, dim=-1)

        return all_h, all_r, target, edges

class TestDataset(Dataset):
    def __init__(self, args, kg, mode):
        self.args = args
        self.kg = kg

        if mode == "valid":
            facts = self.kg.valid_facts
        elif mode == "test":
            facts = self.kg.test_facts
        else:
            raise ValueError

        r2instances = [[] for r in range(self.kg.relation_size)]
        for h, r, t in facts:
            r2instances[r].append((h, r, t))

        self.batches = list()
        for r, instances in enumerate(r2instances):
            for k in range(0, len(instances), args.predictor_batch_size):
                start = k
                end = min(k + args.predictor_batch_size, len(instances))
                self.batches.append(instances[start:end])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def get_batch(self, idx):
        #idx = random.randint(0, len(self.batches) - 1)
        data = self.batches[idx]

        all_h = torch.LongTensor([_[0] for _ in data])
        all_r = torch.LongTensor([_[1] for _ in data])
        all_t = torch.LongTensor([_[2] for _ in data])

        mask = torch.ones(len(data), self.kg.entity_size).bool()
        for k, (h, r, t) in enumerate(data):
            hr_index = self.kg.encode_hr(h, r)
            t_index = torch.LongTensor(self.kg.hr2v[hr_index])
            mask[k][t_index] = 0

        return all_h, all_r, all_t, mask

class RuleDataset(Dataset):
    def __init__(self, num_relations, relation2rules=None):
        self.rules = list()
        self.num_relations = num_relations
        self.ending_idx = num_relations
        self.padding_idx = num_relations + 1
        if relation2rules != None:
            for rules in relation2rules:
                for rule in rules:
                    rule_len = len(rule) - 1
                    formatted_rule = [[rule[k] for k in range(rule_len + 1)] + [self.ending_idx], self.padding_idx, float(rule[-1])]
                    self.rules.append(formatted_rule)

    def __len__(self):
        return len(self.rules)

    def __getitem__(self, idx):
        return self.rules[idx]

    @staticmethod
    def collate_fn(data):
        inputs = [item[0][0:len(item[0])-1] for item in data]
        target = [item[0][1:len(item[0])] for item in data]
        weight = [float(item[-1]) for item in data]
        max_len = max([len(_) for _ in inputs])
        padding_index = [int(item[-2]) for item in data]

        for k in range(len(data)):
            for i in range(max_len - len(inputs[k])):
                inputs[k].append(padding_index[k])
                target[k].append(padding_index[k])

        inputs = torch.LongTensor(inputs)
        target = torch.LongTensor(target)
        weight = torch.Tensor(weight)
        mask = (target != torch.LongTensor(padding_index).unsqueeze(1))

        return inputs, target, mask, weight

def Iterator(dataloader):
    while True:
        for data in dataloader:
            yield data
