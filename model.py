import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch_sparse import spmm
from dataloader import TrainDataset, TestDataset, RuleDataset, Iterator
import rnnlogic_ext

class ReasoningPredictor(torch.nn.Module):
    def __init__(self, args, kg, all_rules):
        super(ReasoningPredictor, self).__init__()
        self.args = args
        self.kg = kg

        self.rules = all_rules
        print("Number of rules:", len(self.rules))

        self.num_entities = kg.entity_size
        self.num_relations = kg.relation_size
        self.num_rules = len(self.rules)

        self.rule_weights = nn.parameter.Parameter(torch.zeros(self.num_rules))

        self.cuda()

    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0

        if self.args.remove_edges == True:
            updated_adjacency = self.kg.get_updated_adjacency(query_r, edges_to_remove)
        else:
            updated_adjacency = None

        score = torch.zeros(self.kg.entity_size, all_r.size(0)).cuda()
        mask = torch.zeros(self.kg.entity_size, all_r.size(0)).cuda()
        for k, (r_head, r_body) in enumerate(self.rules):
            if r_head != query_r:
                continue
            x = self.kg.grounding(all_h, r_head, r_body, updated_adjacency)
            score += x * self.rule_weights[k]
            mask += x

        #print(score.size(), dest.size())
        score = (score / self.args.predictor_temperature).transpose(0, 1)
        
        mask = (mask != 0).transpose(0, 1)
        score = score.masked_fill(~mask, float('-inf'))
        
        return score, mask

    def compute_H(self, all_h, all_r, edges_to_remove, target):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0

        if self.args.remove_edges == True:
            updated_adjacency = self.kg.get_updated_adjacency(query_r, edges_to_remove)
        else:
            updated_adjacency = None

        rule_score = list()
        rule_index = list()
        mask = torch.zeros(self.kg.entity_size, all_r.size(0)).cuda()
        for k, (r_head, r_body) in enumerate(self.rules):
            if r_head != query_r:
                continue
            x = self.kg.grounding(all_h, r_head, r_body, updated_adjacency)
            score = x * self.rule_weights[k]
            mask += x

            rule_score.append(score)
            rule_index.append(k)

        rule_index = torch.LongTensor(rule_index).cuda()
        pos_index = target.transpose(0, 1).bool()
        neg_index = (mask != 0)

        rule_H_score = list()
        for score in rule_score:
            neg_score = (score * neg_index).sum(0) / torch.clamp(neg_index.sum(0), min=1)
            score = score - neg_score.unsqueeze(0)
            H_score = score[pos_index]
            #print(torch.isnan(H_score).sum())
            rule_H_score.append(H_score.unsqueeze(-1))

        rule_H_score = torch.cat(rule_H_score, dim=-1)
        rule_H_score = torch.softmax(rule_H_score, dim=-1).sum(0)

        return rule_H_score, rule_index

    def train_model(self, dataset):
        optim = torch.optim.Adam(self.parameters(), lr=self.args.predictor_learning_rate, weight_decay=self.args.predictor_weight_decay)

        self.train()
        dataset.make_batches()
        total_loss = 0.0
        total_size = 0.0

        for step in range(min(self.args.max_steps_per_iter, len(dataset))):
            all_h, all_r, target, edges_to_remove = dataset.get_batch(step)

            all_h = all_h.cuda()
            all_r = all_r.cuda()
            target = target.cuda()
            edges_to_remove = edges_to_remove.cuda()

            logits, mask = self.forward(all_h, all_r, edges_to_remove)
            if mask.sum().item() == 0:
                continue

            logits = (torch.softmax(logits, dim=1) + 1e-8).log()
            loss = -(logits[mask] * target[mask]).sum() / torch.clamp(target[mask].sum(), min=1)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            total_size += mask.sum().item()

            if (step + 1) % self.args.print_every == 0:
                print('{} {} {:.6f} {:.1f}'.format(step + 1, len(dataset), total_loss / self.args.print_every, total_size / self.args.print_every))
                total_loss = 0.0
                total_size = 0.0

    @torch.no_grad()
    def H_score(self, dataset):
        self.eval()
        dataset.make_batches()

        all_H_score = torch.zeros(self.num_rules).cuda()
        for step in range(len(dataset)):
            all_h, all_r, target, edges_to_remove = dataset.get_batch(step)

            all_h = all_h.cuda()
            all_r = all_r.cuda()
            target = target.cuda()
            edges_to_remove = edges_to_remove.cuda()

            H, index = self.compute_H(all_h, all_r, edges_to_remove, target)
            all_H_score[index] += H / len(dataset)

            if (step + 1) % self.args.print_every == 0:
                print('{} {}'.format(step + 1, len(dataset)))

        r2index_score = [[] for r in range(self.num_relations)]
        for k, rule in enumerate(self.rules):
            index = k
            score = all_H_score[k].item()
            r = rule[0]
            r2index_score[r] += [(index, score)]

        relation2rules = [[] for r in range(self.num_relations)]
        for r in range(self.num_relations):
            r2index_score[r] = sorted(r2index_score[r], key=lambda x:x[1], reverse=True)
            for index, score in r2index_score[r]:
                rule = [self.rules[index][0]] + self.rules[index][1] + [score]
                relation2rules[r].append(rule)

        return relation2rules

    @torch.no_grad()
    def evaluate(self, dataset):
        self.eval()
        ranks = []
        for step in range(len(dataset)):
            all_h, all_r, all_t, flag = dataset.get_batch(step)

            all_h = all_h.cuda()
            all_r = all_r.cuda()
            all_t = all_t.cuda()
            flag = flag.cuda()

            logits, mask = self(all_h, all_r, None)

            for k in range(all_r.size(0)):
                t = all_t[k]
                if mask[k, t].item() == True:
                    val = logits[k, t]
                    rank = (logits[k][flag[k]] >= val).sum().item() + 1
                    ranks += [rank]
                else:
                    ranks += [dataset.kg.entity_size]

        hit1, hit3, hit10, mr, mrr = 0.0, 0.0, 0.0, 0.0, 0.0
        for rank in ranks:
            if rank <= 1:
                hit1 += 1
            if rank <= 3:
                hit3 += 1
            if rank <= 10:
                hit10 += 1
            mr += rank
            mrr += 1.0 / rank

        hit1 /= len(ranks)
        hit3 /= len(ranks)
        hit10 /= len(ranks)
        mr /= len(ranks)
        mrr /= len(ranks)
    
        print('Evaluation: ', len(ranks))
        print('{:.6f}'.format(hit1))
        print('{:.6f}'.format(hit3))
        print('{:.6f}'.format(hit10))
        print('{:.6f}'.format(mr))
        print('{:.6f}'.format(mrr))

class RuleGenerator(torch.nn.Module):
    def __init__(self, num_relations, num_layers, embedding_dim, hidden_dim, cuda=True):
        super(RuleGenerator, self).__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = self.num_relations + 2
        self.label_size = self.num_relations + 1
        self.ending_idx = num_relations
        self.padding_idx = self.num_relations + 1
        self.num_layers = num_layers
        self.use_cuda = cuda

        self.embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)
        self.rnn = torch.nn.LSTM(self.embedding_dim * 2, self.hidden_dim, self.num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.hidden_dim, self.label_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if cuda:
            self.cuda()

    def zero_state(self, batch_size): 
        state_shape = (self.num_layers, batch_size, self.hidden_dim)
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return (h0.cuda(), c0.cuda())
        else:
            return (h0, c0)

    def forward(self, inputs, relation, hidden):
        embedding = self.embedding(inputs)
        embedding_r = self.embedding(relation).unsqueeze(1).expand(-1, inputs.size(1), -1)
        embedding = torch.cat([embedding, embedding_r], dim=-1)
        outputs, hidden = self.rnn(embedding, hidden)
        logits = self.linear(outputs)
        return logits, hidden

    def loss(self, inputs, target, mask, weight):
        if self.use_cuda:
            inputs = inputs.cuda()
            target = target.cuda()
            mask = mask.cuda()
            weight = weight.cuda()

        hidden = self.zero_state(inputs.size(0))
        logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
        logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, self.label_size)
        target = torch.masked_select(target, mask)
        weight = torch.masked_select((mask.t() * weight).t(), mask)
        loss = (self.criterion(logits, target) * weight).sum() / weight.sum()
        return loss

    def log_probability(self, rules):
        if rules == []:
            return []
        with torch.no_grad():
            rules = [rule + [self.ending_idx] for rule in rules]
            max_len = max([len(rule) for rule in rules])
            for k in range(len(rules)):
                rule_len = len(rules[k])
                for i in range(max_len - rule_len):
                    rules[k] += [self.padding_idx]
            rules = torch.LongTensor(rules)
            if self.use_cuda:
                rules = rules.cuda()
            inputs = rules[:, :-1]
            target = rules[:, 1:]
            n, l = target.size(0), target.size(1)
            mask = (target != self.padding_idx)
            hidden = self.zero_state(inputs.size(0))
            logits, hidden = self.forward(inputs, inputs[:, 0], hidden)
            logits = torch.log_softmax(logits, -1)
            logits = logits * mask.unsqueeze(-1)
            target = (target * mask).unsqueeze(-1)
            log_prob = torch.gather(logits, -1, target).squeeze(-1) * mask
            log_prob = log_prob.sum(-1)
        return log_prob.data.cpu().numpy().tolist()
        
    def sample(self, relation, num_samples, max_len, temperature=1.0):
        with torch.no_grad():
            rules = torch.zeros([num_samples, max_len + 1]).long() + self.ending_idx
            log_probabilities = torch.zeros([num_samples, max_len + 1])
            head = torch.LongTensor([relation for k in range(num_samples)])
            if self.use_cuda:
                rules = rules.cuda()
                log_probabilities = log_probabilities.cuda()
                head = head.cuda()

            rules[:, 0] = relation
            hidden = self.zero_state(num_samples)

            for pst in range(max_len):
                inputs = rules[:, pst].unsqueeze(-1)
                if self.use_cuda:
                    inputs = inputs.cuda()
                logits, hidden = self.forward(inputs, head, hidden)
                logits *= temperature
                log_probability = torch.log_softmax(logits.squeeze(1), dim=-1)
                probability = torch.softmax(logits.squeeze(1), dim=-1)
                sample = torch.multinomial(probability, 1)
                log_probability = log_probability.gather(1, sample)

                mask = (rules[:, pst] != self.ending_idx)
                
                rules[mask, pst + 1] = sample.squeeze(-1)[mask]
                log_probabilities[mask, pst + 1] = log_probability.squeeze(-1)[mask]

            length = (rules != self.ending_idx).sum(-1).unsqueeze(-1) - 1
            formatted_rules = torch.cat([length, rules], dim=1)

            log_probabilities = log_probabilities.sum(-1)

        formatted_rules = formatted_rules.data.cpu().numpy().tolist()
        log_probabilities = log_probabilities.data.cpu().numpy().tolist()
        for k in range(num_samples):
            formatted_rules[k].append(log_probabilities[k])

        rule_set = set([tuple(rule) for rule in formatted_rules])
        formatted_rules = [list(rule) for rule in rule_set]

        return formatted_rules

    def next_relation_log_probability(self, seq):
        inputs = torch.LongTensor([seq])
        relation = torch.LongTensor([seq[0]])
        if self.use_cuda:
            inputs = inputs.cuda()
            relation = relation.cuda()
        hidden = self.zero_state(1)
        logits, hidden = self.forward(inputs, relation, hidden)
        log_prob = torch.log_softmax(logits[0, -1, :] * 5, dim=-1).data.cpu().numpy().tolist()
        return log_prob

    def train_model(self, iterator, num_epoch=10000, lr=1e-3, print_epoch=100):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epoch, eta_min=lr/10)

        cum_loss = 0
        for epoch in range(1, num_epoch + 1):
            epoch += 1

            batch = next(iterator)
            inputs, target, mask, weight = batch

            loss = self.loss(inputs, target, mask, weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            cum_loss += loss.item()

            if epoch % print_epoch == 0:
                lr_str = "%.2e" % (opt.param_groups[0]['lr'])
                print(f"train_generator #{epoch} lr = {lr_str} loss = {cum_loss / print_epoch}")
                cum_loss = 0
            if epoch == num_epoch:
                break

    def beam_search(self, relation, num_samples, max_len):
        max_len += 1
        with torch.no_grad():
            found_rules = []
            prev_rules = [[[relation], 0]]
            for k in range(max_len):
                print(f"k = {k} |prev| = {len(prev_rules)}")
                current_rules = list()
                for _i, (rule, score) in enumerate(prev_rules):
                    assert rule[-1] != self.ending_idx
                    log_prob = self.next_relation_log_probability(rule)
                    for i in (range(self.label_size) if (k + 1) != max_len else [self.ending_idx]):
                        new_rule = rule + [i]
                        new_score = score + log_prob[i]
                        (current_rules if i != self.ending_idx else found_rules).append((new_rule, new_score))
                    
                prev_rules = sorted(current_rules, key=lambda x:x[1], reverse=True)[:num_samples]
                found_rules = sorted(found_rules, key=lambda x:x[1], reverse=True)[:num_samples]

            print(f"beam_search |rules| = {len(found_rules)}")
            ret = [[len(rule) - 2] + rule[0:-1] + [score] for rule, score in found_rules]
            return ret

class RNNLogic:
    def __init__(self, args, graph):
        self.args = args
        self.graph = graph
        self.num_relations = graph.relation_size
        self.predictor = None #ReasoningPredictor(self.args, self.graph)
        self.generator = RuleGenerator(self.num_relations, args.generator_layers, args.generator_embedding_dim, args.generator_hidden_dim, args.cuda)

        self.train_dataset = TrainDataset(args, graph)
        self.test_dataset = TestDataset(args, graph, "test")

    # Generate logic rules by sampling.
    def generate_rules(self):
        relation2rules = list()
        for r in range(self.num_relations):
            rules = self.generator.sample(r, self.args.num_generated_rules, self.args.max_rule_length)
            relation2rules.append(rules)
        return relation2rules

    # Generate optimal logic rules by beam search.
    def generate_best_rules(self):
        relation2rules = list()
        for r in range(self.num_relations):
            rules = self.generator.beam_search(r, self.args.num_rules_for_test, self.args.max_rule_length)
            relation2rules.append(rules)
        return relation2rules

    # Update the reasoning predictor with generated logic rules.
    def update_predictor(self, relation2rules):
        all_rules = list()
        for relation, rules in enumerate(relation2rules):
            for rule in rules:
                length = rule[0]
                rule = [rule[1], [rule[2 + k] for k in range(length)]]
                all_rules.append(rule)
        self.predictor = ReasoningPredictor(self.args, self.graph, all_rules)
        self.predictor.train_model(self.train_dataset)
        #self.predictor.set_logic_rules(relation2rules)
        #self.predictor.train(self.args.predictor_learning_rate, self.args.predictor_weight_decay, self.args.predictor_temperature, self.args.predictor_portion, self.args.num_threads)

    # E-step: Infer the high-quality logic rules.
    def e_step(self):
        relation2rules = self.predictor.H_score(self.train_dataset)
        return relation2rules
        #return self.predictor.get_logic_rules()

    # M-step: Update the rule generator with logic rules.
    def m_step(self, relation2rules, tune=False):
        dataset = RuleDataset(self.num_relations, relation2rules)
        dataloader = DataLoader(dataset, batch_size=self.args.generator_batch_size, shuffle=True, num_workers=1, collate_fn=RuleDataset.collate_fn)
        iterator = Iterator(dataloader)
        if not tune:
            self.generator.train_model(iterator, num_epoch=self.args.generator_epochs, lr=self.args.generator_learning_rate)
        else:
            self.generator.train_model(iterator, num_epoch=self.args.generator_tune_epochs, lr=self.args.generator_tune_learning_rate)

    def train(self):
        all_high_quality_rules = [[] for r in range(self.num_relations)]
        
        for iteration in range(self.args.iterations):

            # Generate a set of logic rules and update the reasoning predictor for reasoning.
            relation2rules = self.generate_rules()
            self.update_predictor(relation2rules)
            self.predictor.evaluate(self.test_dataset)
            #print("Valid | MR: {:.6f}, MRR: {:.6f}, Hit@1: {:.6f}, Hit@3: {:.6f}, Hit@10: {:.6f}.".format(mr, mrr, hit1, hit3, hit10))

            # E-step: Identify a subset of high-quality logic rules based on posterior inference.
            high_quality_rules = self.e_step()

            # M-step: Improve the rule generator with the high-quality rules from the E-step.
            self.m_step(high_quality_rules, tune=True)

            for r in range(self.num_relations):
                all_high_quality_rules[r] += high_quality_rules[r]

        self.m_step(all_high_quality_rules)

    def evaluate(self):
        relation2rules = self.generate_best_rules()
        #self.predictor.set_logic_rules(relation2rules)
        #self.predictor.train(self.args.predictor_learning_rate, self.args.predictor_weight_decay, self.args.predictor_temperature, self.args.predictor_portion, self.args.num_threads)
        #mr, mrr, hit1, hit3, hit10 = self.predictor.evaluate("test", self.args.num_threads)
        self.update_predictor(relation2rules)
        self.predictor.evaluate(self.test_dataset)
        #print("Test | MR: {:.6f}, MRR: {:.6f}, Hit@1: {:.6f}, Hit@3: {:.6f}, Hit@10: {:.6f}.".format(mr, mrr, hit1, hit3, hit10))
