import os

def pipeline():
    os.chdir("./mlp")
    os.system("python3 ablation.py")
    os.chdir("../cnn")
    os.system("python3 ablation.py")

if __name__ == "__main__":
    pipeline()