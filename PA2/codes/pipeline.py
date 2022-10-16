import os

def pipeline():
    os.chdir("./mlp")
    os.system("python3 pipeline.py")
    os.chdir("../cnn")
    os.system("python3 pipeline.py")

if __name__ == "__main__":
    pipeline()