import yaml
import os
import sys
import subprocess
import time
import argparse
import csv
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests


class ExperimentTracker:
    def __init__(self, config: Dict):
        self.config = config
        self.lock = threading.Lock()
        self.results = {}  # pattern -> list of results
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.csv_file = None
        self.summary_csv_file = None
        self.notification_config = config.get('notification', {})
        self._init_csv()

    def _init_csv(self):
        log_dir = Path(self.config['experiment'].get('log_dir', './logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = log_dir / f"experiment_results_{timestamp}.csv"
        self.summary_csv_file = log_dir / f"experiment_summary_{timestamp}.csv"
        
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['数据集编号', '重复次数', '图像级AUCROC', '像素级AUCROC', '训练使用时间(秒)', '状态', 'GPU', '完成时间'])
        
        print(f"📁 实验结果将保存到: {self.csv_file}")
        print(f"📊 汇总结果将保存到: {self.summary_csv_file}")

    def update_result(self, pattern: int, status: str, gpu_id: int, 
                     img_roc_auc: Optional[float] = None, 
                     pixel_roc_auc: Optional[float] = None,
                     training_time: Optional[float] = None,
                     error: Optional[str] = None,
                     repeat_idx: int = 1):
        with self.lock:
            if pattern not in self.results:
                self.results[pattern] = []
            
            result = {
                'status': status,
                'gpu_id': gpu_id,
                'img_roc_auc': img_roc_auc,
                'pixel_roc_auc': pixel_roc_auc,
                'training_time': training_time,
                'error': error,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'repeat_idx': repeat_idx
            }
            self.results[pattern].append(result)
            
            if status == 'completed':
                self.completed += 1
            elif status == 'failed':
                self.failed += 1
            
            self._write_to_csv(pattern, result)

    def _write_to_csv(self, pattern: int, result: Dict):
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                pattern,
                result['repeat_idx'],
                f"{result['img_roc_auc']:.4f}" if result['img_roc_auc'] is not None else "N/A",
                f"{result['pixel_roc_auc']:.4f}" if result['pixel_roc_auc'] is not None else "N/A",
                f"{result['training_time']:.2f}" if result['training_time'] is not None else "N/A",
                result['status'],
                result['gpu_id'],
                result['timestamp']
            ])

    def print_summary(self):
        print("\n" + "="*80)
        print("📊 EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Total tasks: {self.total}")
        print(f"Completed: {self.completed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success rate: {self.completed/self.total*100:.1f}%")
        print(f"\n📁 结果已保存到: {self.csv_file}")
        
        print("\n" + "="*80)
        print("📋 实验结果表格")
        print("="*80)
        print(f"{'数据集编号':<12} {'重复次数':<8} {'图像级AUCROC':<15} {'像素级AUCROC':<15} {'训练时间(秒)':<15} {'状态':<10}")
        print("-"*80)
        
        for pattern in sorted(self.results.keys()):
            for result in self.results[pattern]:
                img_auc = f"{result['img_roc_auc']:.4f}" if result['img_roc_auc'] is not None else "N/A"
                pixel_auc = f"{result['pixel_roc_auc']:.4f}" if result['pixel_roc_auc'] is not None else "N/A"
                training_time = f"{result['training_time']:.2f}" if result['training_time'] is not None else "N/A"
                status_icon = "✅" if result['status'] == 'completed' else "❌"
                print(f"{pattern:<12} {result['repeat_idx']:<8} {img_auc:<15} {pixel_auc:<15} {training_time:<15} {status_icon} {result['status']:<8}")
        
        print("\n" + "="*80)
        print("🏆 最优结果和平均值")
        print("="*80)
        print(f"{'数据集编号':<12} {'最优图像级AUC':<15} {'最优像素级AUC':<15}")
        print("-"*80)
        
        # 计算最优结果和平均值
        all_img_aucs = []
        all_pixel_aucs = []
        best_results = {}
        
        for pattern in sorted(self.results.keys()):
            completed_results = [r for r in self.results[pattern] if r['status'] == 'completed' and r['img_roc_auc'] is not None]
            if completed_results:
                # 找到最优结果（最大的图像级AUC）
                best_result = max(completed_results, key=lambda x: x['img_roc_auc'])
                best_results[pattern] = best_result
                
                img_auc = f"{best_result['img_roc_auc']:.4f}"
                pixel_auc = f"{best_result['pixel_roc_auc']:.4f}"
                print(f"{pattern:<12} {img_auc:<15} {pixel_auc:<15}")
                
                all_img_aucs.append(best_result['img_roc_auc'])
                all_pixel_aucs.append(best_result['pixel_roc_auc'])
        
        # 计算平均值
        if all_img_aucs:
            avg_img_auc = sum(all_img_aucs) / len(all_img_aucs)
            avg_pixel_auc = sum(all_pixel_aucs) / len(all_pixel_aucs)
            print("-"*80)
            print(f"{'平均值':<12} {avg_img_auc:.4f}          {avg_pixel_auc:.4f}")
        
        # 写入汇总 CSV
        with open(self.summary_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['数据集编号', '最优图像级AUCROC', '最优像素级AUCROC'])
            for pattern, result in best_results.items():
                writer.writerow([
                    pattern,
                    f"{result['img_roc_auc']:.4f}",
                    f"{result['pixel_roc_auc']:.4f}"
                ])
            if all_img_aucs:
                writer.writerow(['平均值', f"{avg_img_auc:.4f}", f"{avg_pixel_auc:.4f}"])
        
        print(f"\n📊 汇总结果已保存到: {self.summary_csv_file}")
        print("="*80 + "\n")

    def send_notification(self, pattern: int = None, img_auc: float = None, pixel_auc: float = None):
        if not self.notification_config.get('enabled', False):
            return
        
        try:
            url = self.notification_config.get('url', '')
            timeout = self.notification_config.get('timeout', 5)
            
            if pattern is not None:
                data = {
                    'type': 'training_complete',
                    'pattern': pattern,
                    'img_auc': f"{img_auc:.4f}" if img_auc is not None else "N/A",
                    'pixel_auc': f"{pixel_auc:.4f}" if pixel_auc is not None else "N/A"
                }
            else:
                data = {
                    'type': 'all_complete',
                    'total': self.total,
                    'completed': self.completed,
                    'failed': self.failed
                }
            
            response = requests.post(url, json=data, timeout=timeout)
            if response.status_code == 200:
                print("✅ 通知发送成功")
            else:
                print(f"⚠️  通知发送失败，状态码: {response.status_code}")
        except requests.exceptions.Timeout:
            print("⚠️  通知发送超时")
        except requests.exceptions.ConnectionError:
            print("⚠️  无法连接到通知服务器")
        except Exception as e:
            print(f"⚠️  通知发送失败: {e}")


class GPUPool:
    def __init__(self, gpu_ids: List[int], max_concurrent_per_gpu: int = 1):
        self.gpu_ids = gpu_ids
        self.max_concurrent = max_concurrent_per_gpu
        self.available_gpus = {gpu_id: max_concurrent_per_gpu for gpu_id in gpu_ids}
        self.lock = threading.Lock()

    def acquire(self) -> Optional[int]:
        with self.lock:
            for gpu_id in self.gpu_ids:
                if self.available_gpus[gpu_id] > 0:
                    self.available_gpus[gpu_id] -= 1
                    return gpu_id
            return None

    def release(self, gpu_id: int):
        with self.lock:
            if gpu_id in self.available_gpus:
                self.available_gpus[gpu_id] += 1


def run_single_experiment(pattern: int, gpu_id: int, config: Dict, 
                          tracker: ExperimentTracker, retry_count: int = 0, repeat_idx: int = 1) -> bool:
    experiment_name = config['experiment']['name_template'].format(
        epochs=config['training']['num_epochs'],
        pattern=pattern,
        repeat=repeat_idx
    )
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Pattern {pattern} on GPU {gpu_id}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Retry: {retry_count}/{config['scheduler']['max_retries']}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, 'b_run.py',
        '--phase', 'train',
        '--num_of_pattern', str(pattern),
        '--num_of_epochs', str(config['training']['num_epochs']),
        '--early_stop_patience', str(config['training']['early_stop_patience']),
        '--experiment_name', experiment_name,
        '--gpu_id', str(gpu_id),
        '--repeat', str(repeat_idx)
    ]

    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config['scheduler']['timeout'] * 60
        )
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ Pattern {pattern} completed successfully in {elapsed_time/60:.1f} minutes")
            
            img_auc = None
            pixel_auc = None
            for line in result.stdout.split('\n'):
                if 'image ROCAUC:' in line:
                    try:
                        img_auc = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'pixel ROCAUC:' in line:
                    try:
                        pixel_auc = float(line.split(':')[1].strip())
                    except:
                        pass
            
            tracker.update_result(
                pattern=pattern,
                status='completed',
                gpu_id=gpu_id,
                img_roc_auc=img_auc,
                pixel_roc_auc=pixel_auc,
                training_time=elapsed_time,
                repeat_idx=repeat_idx
            )
            
            tracker.send_notification(pattern, img_auc, pixel_auc)
            return True
        else:
            print(f"\n❌ Pattern {pattern} failed with return code {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            
            if retry_count < config['scheduler']['max_retries']:
                print(f"🔄 Retrying Pattern {pattern}...")
                time.sleep(5)
                return run_single_experiment(pattern, gpu_id, config, tracker, retry_count + 1)
            else:
                tracker.update_result(
                    pattern=pattern,
                    status='failed',
                    gpu_id=gpu_id,
                    error=result.stderr[:500],
                    repeat_idx=repeat_idx
                )
                return False

    except subprocess.TimeoutExpired:
        print(f"\n⏰ Pattern {pattern} timed out after {config['scheduler']['timeout']} minutes")
        tracker.update_result(
            pattern=pattern,
            status='failed',
            gpu_id=gpu_id,
            error=f"Timeout after {config['scheduler']['timeout']} minutes",
            repeat_idx=repeat_idx
        )
        return False
    except Exception as e:
        print(f"\n❌ Pattern {pattern} failed with exception: {str(e)}")
        tracker.update_result(
            pattern=pattern,
            status='failed',
            gpu_id=gpu_id,
            error=str(e),
            repeat_idx=repeat_idx
        )
        return False


def run_experiments_sequential(config: Dict, tracker: ExperimentTracker, gpu_pool: GPUPool):
    patterns = config['dataset']['patterns']
    repeat_times = config['dataset'].get('repeat_times', 1)
    tracker.total = len(patterns) * repeat_times

    for pattern in patterns:
        for repeat_idx in range(1, repeat_times + 1):
            gpu_id = gpu_pool.acquire()
            while gpu_id is None:
                print("⏳ Waiting for available GPU...")
                time.sleep(5)
                gpu_id = gpu_pool.acquire()

            try:
                success = run_single_experiment(pattern, gpu_id, config, tracker, repeat_idx=repeat_idx)
                if not success and not config['scheduler']['continue_on_failure']:
                    print("❌ Stopping due to failure and continue_on_failure=False")
                    return
            finally:
                gpu_pool.release(gpu_id)


def run_experiments_parallel(config: Dict, tracker: ExperimentTracker, gpu_pool: GPUPool):
    task_queue = queue.Queue()
    repeat_times = config['dataset'].get('repeat_times', 1)
    for pattern in config['dataset']['patterns']:
        for repeat_idx in range(1, repeat_times + 1):
            task_queue.put((pattern, repeat_idx))
    tracker.total = task_queue.qsize()
    
    def worker():
        while True:
            gpu_id = gpu_pool.acquire()
            if gpu_id is None:
                break
            
            try:
                try:
                    task = task_queue.get(block=False)
                except queue.Empty:
                    gpu_pool.release(gpu_id)
                    break
                pattern, repeat_idx = task
                
                success = run_single_experiment(pattern, gpu_id, config, tracker, repeat_idx=repeat_idx)
                if not success and not config['scheduler']['continue_on_failure']:
                    # 清空队列
                    while not task_queue.empty():
                        try:
                            task_queue.get(block=False)
                        except:
                            pass
            finally:
                gpu_pool.release(gpu_id)
    
    num_workers = len(config['gpu']['devices']) * config['gpu']['max_concurrent_per_gpu']
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        for future in as_completed(futures):
            future.result()


def run_experiments_round_robin(config: Dict, tracker: ExperimentTracker, gpu_pool: GPUPool):
    tasks = []
    repeat_times = config['dataset'].get('repeat_times', 1)
    for pattern in config['dataset']['patterns']:
        for repeat_idx in range(1, repeat_times + 1):
            tasks.append((pattern, repeat_idx))
    gpu_ids = config['gpu']['devices']
    tracker.total = len(tasks)
    
    for i, task in enumerate(tasks):
        pattern, repeat_idx = task
        gpu_id = gpu_ids[i % len(gpu_ids)]
        success = run_single_experiment(pattern, gpu_id, config, tracker, repeat_idx=repeat_idx)
        if not success and not config['scheduler']['continue_on_failure']:
            print("❌ Stopping due to failure and continue_on_failure=False")
            break


def main():
    parser = argparse.ArgumentParser(description='Experiment Training Manager')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--patterns', type=str, nargs='+', help='Override patterns to train')
    parser.add_argument('--gpu', type=int, nargs='+', help='Override GPU devices')
    parser.add_argument('--mode', type=str, default='round_robin', 
                       choices=['sequential', 'parallel', 'round_robin'],
                       help='Execution mode')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.patterns:
        config['dataset']['patterns'] = [int(p) for p in args.patterns]
    if args.gpu:
        config['gpu']['devices'] = args.gpu
    if args.mode:
        config['scheduler']['execution_mode'] = args.mode

    print("\n" + "="*80)
    print("🔧 EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Patterns to train: {config['dataset']['patterns']}")
    print(f"GPU devices: {config['gpu']['devices']}")
    print(f"Execution mode: {config['scheduler']['execution_mode']}")
    print(f"Max retries: {config['scheduler']['max_retries']}")
    print(f"Timeout: {config['scheduler']['timeout']} minutes")
    print(f"Continue on failure: {config['scheduler']['continue_on_failure']}")
    print("="*80 + "\n")

    if args.dry_run:
        print("🔍 DRY RUN MODE - Commands will not be executed\n")
        for pattern in config['dataset']['patterns']:
            gpu_id = config['gpu']['devices'][pattern % len(config['gpu']['devices'])]
            experiment_name = config['experiment']['name_template'].format(
                epochs=config['training']['num_epochs'],
                pattern=pattern
            )
            print(f"Pattern {pattern}: GPU {gpu_id}, Experiment: {experiment_name}")
        return

    tracker = ExperimentTracker(config)
    gpu_pool = GPUPool(
        config['gpu']['devices'],
        config['gpu']['max_concurrent_per_gpu']
    )

    start_time = time.time()
    
    execution_mode = config['scheduler']['execution_mode']
    if execution_mode == 'sequential':
        run_experiments_sequential(config, tracker, gpu_pool)
    elif execution_mode == 'parallel':
        run_experiments_parallel(config, tracker, gpu_pool)
    elif execution_mode == 'round_robin':
        run_experiments_round_robin(config, tracker, gpu_pool)

    elapsed_time = time.time() - start_time

    tracker.print_summary()
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    
    tracker.send_notification()


if __name__ == '__main__':
    main()
