import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', default='cora')
args.add_argument('--model', default='gcn')
args.add_argument('--learning_rate', default=0.01)
args.add_argument('--epochs', default=200)
args.add_argument('--hidden1', default=16)
args.add_argument('--dropout', default=0.5)
args.add_argument('--weight_decay', default=5e-4)
args.add_argument('--early_stopping', default=10)
args.add_argument('--max_degree', default=3)


args = args.parse_args()
print(args)