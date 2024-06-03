import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import argparse

parser = argparse.ArgumentParser(description='Correlation between classification loss and pretext task loss')
parser.add_argument('--dataset', type=str, required=True, choices=['Cifar10', 'Caltech101', 'Imbalanced_Cifar10'], help='Dataset')
parser.add_argument('--task', type=str, required=True, choices=['rotation', 'colorization'], help='Task')
args = parser.parse_args()

def load_data(filepath, columns):
    data = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split('_', 1)
            if len(parts) == 2:
                data.append(parts)
    df = pd.DataFrame(data, columns=columns)
    df[columns[0]] = df[columns[0]].astype(float)
    return df

def main():

    classification_loss_path = f'./classification_loss_{args.dataset}.txt'
    task_loss_path = f'./{args.task}_loss_{args.dataset}.txt'

    # 데이터 불러오기
    classification_loss = load_data(classification_loss_path, ["classification_loss", "file_path"])
    task_loss = load_data(task_loss_path, [f"{args.task}_loss", "file_path"])

    # 데이터 병합
    merged_df = pd.merge(classification_loss, task_loss, on='file_path')
    print(merged_df.head())

    # 데이터 랭크 변환
    merged_df['classification_loss_rank'] = merged_df['classification_loss'].rank(pct=True)
    merged_df[f'{args.task}_loss_rank'] = merged_df[f'{args.task}_loss'].rank(pct=True)

    # 상관계수 계산
    correlation, _ = spearmanr(merged_df['classification_loss_rank'], merged_df[f'{args.task}_loss_rank'])

    # 상관관계 시각화 및 저장
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='classification_loss_rank', y=f'{args.task}_loss_rank', data=merged_df, alpha=0.5, s=10)
    plt.title(f'Correlation with Classification Loss and {args.task} Loss {args.dataset} (ρ = {correlation:.2f})')
    plt.xlabel('Classification Loss Rank')
    plt.ylabel(f'{args.task} Loss Rank')
    plt.grid(True)
    plt.savefig(f'Correlation_{args.dataset}_{args.task}.png')
    plt.close()

    # 추가적으로 1000개 샘플 랜덤 추출 후 시각화 및 저장
    sampled_data = merged_df.sample(n=1000, random_state=10)
    correlation, _ = spearmanr(sampled_data['classification_loss_rank'], sampled_data[f'{args.task}_loss_rank'])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='classification_loss_rank', y=f'{args.task}_loss_rank', data=sampled_data, alpha=0.5, s=10)
    plt.title(f'Random Sample Correlation with Classification Loss and {args.task} Loss {args.dataset} (ρ = {correlation:.2f})')
    plt.xlabel('Classification Loss Rank')
    plt.ylabel(f'{args.task} Loss Rank')
    plt.grid(True)
    plt.savefig(f'Random_Correlation_{args.dataset}_{args.task}.png')
    plt.close()

if __name__ == '__main__':
    main()
