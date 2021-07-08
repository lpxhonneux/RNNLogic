import os

from omegaconf import DictConfig, OmegaConf
import hydra

from model import RNNLogic
from dataloader import KnowledgeGraph, TrainDataset, TestDataset
import rnnlogic_ext as pyrnnlogic

@hydra.main(config_path="./conf", config_name="config")
def main(cfg):
    print("Working directory : {}".format(os.getcwd()))
    print("Config:")

    print(OmegaConf.to_yaml(cfg))
    args = cfg
    args.max_steps_per_iter = 10000
    args.remove_edges = True
    args.print_every = 1000

    p_kg = pyrnnlogic.new_knowledge_graph(args.data_path)
    p_rm = pyrnnlogic.new_rule_miner(p_kg)
    pyrnnlogic.run_rule_miner(p_rm, 3, 1.0, 8) # 3 is max rule length, 8 is num of thread.
    rules = pyrnnlogic.get_logic_rules(p_rm)

    graph = KnowledgeGraph(args)
    relation2rules = [[] for r in range(graph.relation_size)]
    for rule in rules:
        relation = rule[0]
        rule = rule + [1.0]
        relation2rules[relation].append(rule)

    rnnlogic = RNNLogic(args, graph)
    rnnlogic.m_step(relation2rules)

    rnnlogic.train()
    rnnlogic.evaluate()

if __name__ == '__main__':
    main()
