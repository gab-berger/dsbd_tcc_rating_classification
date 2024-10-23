import pandas as pd

def main():
    column = 'Likes'
    likes = pd.read_csv('Amazon_Reviews.csv', usecols=[column])[column].tolist()
    for n in range(4):
        print(str(n+1),'-',likes[n])

if __name__ == '__main__':
    main()