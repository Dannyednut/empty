import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data")
args = parser.parse_args()
DATA = args.data

class Out():
    def __init__(self):
        self.data = None
    
    def print_data(self):
        self.data = DATA
        print(self.data)

if __name__ == "__main__":
    out = Out()
    out.print_data()