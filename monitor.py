#!/usr/bin/env python3
"""
训练过程监控脚本
实时显示训练进度和性能指标
"""

import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

class TrainingMonitor:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.log_file = os.path.join(results_dir, "training_log.csv")
        self.loss_file = os.path.join(results_dir, "train_loss.csv")
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not os.path.exists(self.loss_file):
            print(f"❌ 找不到训练日志文件: {self.loss_file}")
            return
            
        try:
            # 读取训练数据
            df = pd.read_csv(self.loss_file)
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('MAFM-AttU_Net 训练监控', fontsize=16)
            
            # 训练损失
            if 'train_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='训练损失')
                axes[0, 0].set_title('训练损失')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # 验证损失
            if 'val_loss' in df.columns:
                axes[0, 1].plot(df['epoch'], df['val_loss'], 'r-', label='验证损失')
                axes[0, 1].set_title('验证损失')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Dice系数
            if 'dice_score' in df.columns:
                axes[1, 0].plot(df['epoch'], df['dice_score'], 'g-', label='Dice系数')
                axes[1, 0].set_title('Dice系数')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Dice Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # IoU
            if 'iou' in df.columns:
                axes[1, 1].plot(df['epoch'], df['iou'], 'm-', label='IoU')
                axes[1, 1].set_title('IoU')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('IoU')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(self.results_dir, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ 训练曲线已保存到: {plot_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 绘制训练曲线失败: {e}")
    
    def show_latest_metrics(self):
        """显示最新的训练指标"""
        if not os.path.exists(self.loss_file):
            print(f"❌ 找不到训练日志文件: {self.loss_file}")
            return
            
        try:
            df = pd.read_csv(self.loss_file)
            if len(df) == 0:
                print("📊 暂无训练数据")
                return
                
            latest = df.iloc[-1]
            
            print("📊 最新训练指标:")
            print(f"Epoch: {latest.get('epoch', 'N/A')}")
            print(f"训练损失: {latest.get('train_loss', 'N/A'):.4f}")
            print(f"验证损失: {latest.get('val_loss', 'N/A'):.4f}")
            print(f"Dice系数: {latest.get('dice_score', 'N/A'):.4f}")
            print(f"IoU: {latest.get('iou', 'N/A'):.4f}")
            print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ 读取训练指标失败: {e}")
    
    def watch_training(self, interval=30):
        """实时监控训练过程"""
        print(f"🔍 开始监控训练过程 (每{interval}秒更新)")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                print("=" * 50)
                print("MAFM-AttU_Net 训练监控")
                print("=" * 50)
                
                self.show_latest_metrics()
                
                print(f"\n⏰ 下次更新: {interval}秒后")
                print("按 Ctrl+C 停止监控")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n✅ 监控已停止")
    
    def generate_report(self):
        """生成训练报告"""
        if not os.path.exists(self.loss_file):
            print(f"❌ 找不到训练日志文件: {self.loss_file}")
            return
            
        try:
            df = pd.read_csv(self.loss_file)
            
            report = f"""
# MAFM-AttU_Net 训练报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 训练概况
- 总训练轮数: {len(df)}
- 最佳验证损失: {df['val_loss'].min():.4f}
- 最佳Dice系数: {df['dice_score'].max():.4f}
- 最佳IoU: {df['iou'].max():.4f}

## 训练稳定性
- 损失标准差: {df['train_loss'].std():.4f}
- 验证损失标准差: {df['val_loss'].std():.4f}

## 模型收敛性
- 最后10轮平均训练损失: {df['train_loss'].tail(10).mean():.4f}
- 最后10轮平均验证损失: {df['val_loss'].tail(10).mean():.4f}
"""
            
            report_path = os.path.join(self.results_dir, 'training_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ 训练报告已保存到: {report_path}")
            print(report)
            
        except Exception as e:
            print(f"❌ 生成训练报告失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="MAFM-AttU_Net 训练监控工具")
    parser.add_argument('--action', choices=['plot', 'watch', 'metrics', 'report'], 
                       default='plot', help='执行的操作')
    parser.add_argument('--results-dir', default='results', help='结果目录路径')
    parser.add_argument('--interval', type=int, default=30, help='监控更新间隔(秒)')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.results_dir)
    
    if args.action == 'plot':
        monitor.plot_training_curves()
    elif args.action == 'watch':
        monitor.watch_training(args.interval)
    elif args.action == 'metrics':
        monitor.show_latest_metrics()
    elif args.action == 'report':
        monitor.generate_report()

if __name__ == "__main__":
    main()