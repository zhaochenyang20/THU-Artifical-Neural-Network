import os

def pipeline():
    os.system("python3 ./mlp/pipeline.py")
    os.system("python3 ./cnn/pipeline.py")

if __name__ == "__main__":
    pipeline()