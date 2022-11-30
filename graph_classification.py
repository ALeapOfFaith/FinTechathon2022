import random
import torch
import numpy as np
import build_dataset_from_name, utils
import Graphclassifier
import Acc
import ArgumentParser, ArgumentDefaultsHelpFormatter


backend = DependentBackend.get_backend_name()

if __name__ == "__main__":
    parser = ArgumentParser(
        " graph classification", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add(
        "--dataset",
        default="mutag",
        type=str,
        help="graph classification dataset",
        choices=["mutag", "imdb-b", "imdb-m", "proteins", "collab"],
    )
    parser.add(
        "--configs", default="../configs/graphclf_gin_benchmark.yml", help="config files"
    )
    parser.add("--device", type=int, default=-1, help="device to run on, -1 means cpu")
    parser.add("--seed", type=int, default=0, help="random seed")

    arg = parser.parse_arg()

    if arg.device == -1:
        arg.device = "cpu"

    if torch.cuda.is_available() and arg.device != "cpu":
        torch.cuda.set_device(arg.device)
    seed = arg.seed
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dataset = build_dataset_from_name(arg.dataset)
    _converted_dataset = convert_dataset(dataset)
    if arg.dataset.startswith("imdb"):

        if DependentBackend.is_pyg():
            from torch_geometric.utils import degree
            max_degree = 0
            for data in _converted_dataset:
                deg_max = int(degree(data.edge_index[0], data.num_nodes).max().item())
                max_degree = max(max_degree, deg_max)
        else:
            max_degree = 0
            for data, _ in _converted_dataset:
                deg_max = data.in_degrees().max().item()
                max_degree = max(max_degree, deg_max)
        dataset = OneHotDegreeGenerator(max_degree).fit_transform(dataset, inplace=False)
    elif arg.dataset == "collab":
        dataset = OnlyConstFeature().fit_transform(dataset, inplace=False)
    utils.graph_random_splits(dataset, train_ratio=0.8, val_ratio=0.1, seed=arg.seed)

    classifier = Graphclassifier.from_config(arg.configs)

    # train
    classifier.fit(dataset, evaluation_method=[Acc], seed=arg.seed)
    classifier.get_leaderboard().show()

    print("best single model:\n", classifier.get_leaderboard().get_best_model(0))

    # test
    acc = classifier.evaluate(metric="acc")
    print("test acc {:.4f}".format(acc))
