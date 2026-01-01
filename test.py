import sys

def out(name):
    print(f"Hello {name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py name")
        sys.exit(1)
    out(sys.argv[1])