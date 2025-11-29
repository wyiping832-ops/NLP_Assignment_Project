import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelBenchmark:
    """Model Performance Benchmarking Framework"""
    
    def __init__(self):
        self.setup_benchmark_framework()
        
    def setup_benchmark_framework(self):
        """Setup benchmark framework"""
        self.models_comparison = {
            'sentiment_analysis': [
                'VADER (rule-based)',
                'TextBlob (lexicon-based)', 
                'BERT-base-uncased',
                'RoBERTa-base',
                'DistilBERT (efficient)'
            ],
            'named_entity_recognition': [
                'spaCy en_core_web_sm',
                'BERT-NER',
                'XLM-RoBERTa-NER'
            ],
            'text_classification': [
                'TF-IDF + SVM',
                'FastText',
                'BERT-base',
                'DistilBERT'
            ]
        }
        
        self.evaluation_metrics = [
            'accuracy',
            'precision', 
            'recall',
            'f1_score',
            'processing_speed',
            'memory_usage',
            'model_size'
        ]
        
        self.test_datasets = [
            'IMDB movie reviews',
            'Twitter sentiment dataset',
            'CoNLL-2003 NER',
            'Custom internet data collection'
        ]
    
    def generate_synthetic_benchmark_data(self):
        """Generate synthetic benchmark data"""
        print("📊 Generating synthetic benchmark data")
        print("=" * 50)
        
        benchmark_results = {}
        
        # Generate performance data for each task
        for task, models in self.models_comparison.items():
            task_results = {}
            
            for model in models:
                performance = {}
                
                # Core performance metrics (simulated realistic ranges)
                if 'VADER' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.65, 0.75),
                        'precision': np.random.uniform(0.60, 0.70),
                        'recall': np.random.uniform(0.65, 0.75),
                        'f1_score': np.random.uniform(0.63, 0.73)
                    })
                elif 'TextBlob' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.60, 0.70),
                        'precision': np.random.uniform(0.58, 0.68),
                        'recall': np.random.uniform(0.62, 0.72),
                        'f1_score': np.random.uniform(0.60, 0.70)
                    })
                elif 'DistilBERT' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.85, 0.92),
                        'precision': np.random.uniform(0.84, 0.91),
                        'recall': np.random.uniform(0.85, 0.92),
                        'f1_score': np.random.uniform(0.85, 0.92)
                    })
                elif 'BERT' in model or 'RoBERTa' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.88, 0.95),
                        'precision': np.random.uniform(0.87, 0.94),
                        'recall': np.random.uniform(0.88, 0.95),
                        'f1_score': np.random.uniform(0.88, 0.95)
                    })
                elif 'spaCy' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.82, 0.88),
                        'precision': np.random.uniform(0.80, 0.86),
                        'recall': np.random.uniform(0.83, 0.89),
                        'f1_score': np.random.uniform(0.82, 0.88)
                    })
                elif 'TF-IDF' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.75, 0.82),
                        'precision': np.random.uniform(0.74, 0.81),
                        'recall': np.random.uniform(0.75, 0.82),
                        'f1_score': np.random.uniform(0.75, 0.82)
                    })
                elif 'FastText' in model:
                    performance.update({
                        'accuracy': np.random.uniform(0.78, 0.85),
                        'precision': np.random.uniform(0.77, 0.84),
                        'recall': np.random.uniform(0.78, 0.85),
                        'f1_score': np.random.uniform(0.78, 0.85)
                    })
                else:
                    performance.update({
                        'accuracy': np.random.uniform(0.70, 0.80),
                        'precision': np.random.uniform(0.68, 0.78),
                        'recall': np.random.uniform(0.71, 0.81),
                        'f1_score': np.random.uniform(0.70, 0.80)
                    })
                
                # Resource usage metrics
                if 'VADER' in model or 'TextBlob' in model:
                    performance.update({
                        'processing_speed': np.random.uniform(800, 1200),
                        'memory_usage': np.random.uniform(50, 100),
                        'model_size': np.random.uniform(1, 5)
                    })
                elif 'DistilBERT' in model:
                    performance.update({
                        'processing_speed': np.random.uniform(80, 150),
                        'memory_usage': np.random.uniform(400, 600),
                        'model_size': np.random.uniform(250, 350)
                    })
                elif 'BERT' in model or 'RoBERTa' in model:
                    performance.update({
                        'processing_speed': np.random.uniform(40, 80),
                        'memory_usage': np.random.uniform(800, 1200),
                        'model_size': np.random.uniform(400, 500)
                    })
                else:
                    performance.update({
                        'processing_speed': np.random.uniform(100, 300),
                        'memory_usage': np.random.uniform(200, 400),
                        'model_size': np.random.uniform(50, 150)
                    })
                
                task_results[model] = performance
            
            benchmark_results[task] = task_results
        
        self.benchmark_data = benchmark_results
        return benchmark_results
    
    def create_performance_visualizations(self):
        """Create performance visualizations"""
        if not hasattr(self, 'benchmark_data'):
            self.generate_synthetic_benchmark_data()
        
        for task, models_data in self.benchmark_data.items():
            print(f"\n📈 Performance comparison for {task}")
            print("=" * 40)
            
            models = list(models_data.keys())
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{task} - Model Performance Benchmark', fontsize=16)
            
            # Accuracy
            accuracies = [models_data[m]['accuracy'] for m in models]
            bars1 = axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim(0, 1)
            
            for bar, acc in zip(bars1, accuracies):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            # F1 Score
            f1_scores = [models_data[m]['f1_score'] for m in models]
            bars2 = axes[0, 1].bar(models, f1_scores, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('F1 Score Comparison')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].set_ylim(0, 1)
            
            for bar, f1 in zip(bars2, f1_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Processing Speed
            speeds = [models_data[m]['processing_speed'] for m in models]
            bars3 = axes[1, 0].bar(models, speeds, color='orange', alpha=0.7)
            axes[1, 0].set_title('Processing Speed (docs/sec)')
            axes[1, 0].set_ylabel('Speed')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for bar, sp in zip(bars3, speeds):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                f'{sp:.0f}', ha='center', va='bottom', fontsize=8)
            
            # Memory Usage
            mem = [models_data[m]['memory_usage'] for m in models]
            bars4 = axes[1, 1].bar(models, mem, color='pink', alpha=0.7)
            axes[1, 1].set_title('Memory Usage (MB)')
            axes[1, 1].set_ylabel('Memory')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for bar, m in zip(bars4, mem):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                f'{m:.0f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.show()
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis report"""
        if not hasattr(self, 'benchmark_data'):
            self.generate_synthetic_benchmark_data()
        
        print("\n📋 Comprehensive Benchmark Analysis Report")
        print("=" * 60)
        
        analysis_results = {}
        
        for task, models_data in self.benchmark_data.items():
            print(f"\n🔍 Analysis for {task}:")
            print("-" * 40)
            
            best_f1 = max(models_data.items(), key=lambda x: x[1]['f1_score'])
            best_speed = max(models_data.items(), key=lambda x: x[1]['processing_speed'])
            best_memory = min(models_data.items(), key=lambda x: x[1]['memory_usage'])
            
            print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['f1_score']:.3f})")
            print(f"Fastest Model: {best_speed[0]} ({best_speed[1]['processing_speed']:.0f} docs/sec)")
            print(f"Lowest Memory Usage: {best_memory[0]} ({best_memory[1]['memory_usage']:.0f} MB)")
            
            efficiency_scores = {}
            for model, metrics in models_data.items():
                efficiency_score = (
                    metrics['f1_score'] * 0.6 +
                    (metrics['processing_speed'] / 1000) * 0.2 +
                    (1 - metrics['memory_usage'] / 1000) * 0.2
                )
                efficiency_scores[model] = efficiency_score
            
            best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1])
            print(f"Most Efficient Model: {best_efficiency[0]} (Score: {best_efficiency[1]:.3f})")
            
            analysis_results[task] = {
                'best_f1': best_f1,
                'best_speed': best_speed,
                'best_memory': best_memory,
                'best_efficiency': best_efficiency,
                'efficiency_scores': efficiency_scores
            }
        
        return analysis_results
    
    def create_recommendation_matrix(self):
        """Create recommendation matrix"""
        if not hasattr(self, 'benchmark_data'):
            self.generate_synthetic_benchmark_data()
        
        print("\n💡 Model Recommendation Matrix")
        print("=" * 50)
        
        recommendations = {
            'High Accuracy Needed': {
                'sentiment_analysis': 'RoBERTa-base',
                'named_entity_recognition': 'BERT-NER', 
                'text_classification': 'BERT-base',
                'reasoning': 'Large pretrained models provide highest accuracy'
            },
            'Real-Time Processing': {
                'sentiment_analysis': 'VADER',
                'named_entity_recognition': 'spaCy en_core_web_sm',
                'text_classification': 'FastText',
                'reasoning': 'Lightweight models offer fastest speed'
            },
            'Resource-Constrained Environment': {
                'sentiment_analysis': 'TextBlob',
                'named_entity_recognition': 'spaCy en_core_web_sm', 
                'text_classification': 'TF-IDF + SVM',
                'reasoning': 'Low memory and storage requirements'
            },
            'Balanced Performance': {
                'sentiment_analysis': 'DistilBERT',
                'named_entity_recognition': 'XLM-RoBERTa-NER',
                'text_classification': 'DistilBERT', 
                'reasoning': 'Good balance between accuracy and efficiency'
            }
        }
        
        recommendation_df = pd.DataFrame(recommendations).T
        print(recommendation_df)
        
        # Heatmap
        plt.figure(figsize=(12, 8))
        
        scenarios = list(recommendations.keys())
        tasks = ['sentiment_analysis', 'named_entity_recognition', 'text_classification']
        
        heatmap_data = []
        for scenario in scenarios:
            row = []
            for task in tasks:
                model = recommendations[scenario][task]
                if 'BERT' in model or 'RoBERTa' in model:
                    score = 3
                elif 'Distil' in model:
                    score = 2
                else:
                    score = 1
                row.append(score)
            heatmap_data.append(row)
        
        sns.heatmap(
            heatmap_data, annot=True, fmt='d',
            xticklabels=[task.replace('_', ' ').title() for task in tasks],
            yticklabels=[s for s in scenarios],
            cmap='YlOrRd'
        )
        
        plt.title('Model Recommendation Matrix (3=High Performance, 2=Balanced, 1=Efficient)')
        plt.tight_layout()
        plt.show()
        
        return recommendation_df


# Run Assignment 3 Demo
print("\n🎯 Assignment 3: Model Performance Benchmarking")
print("=" * 60)

benchmark = ModelBenchmark()

benchmark_data = benchmark.generate_synthetic_benchmark_data()

benchmark.create_performance_visualizations()

analysis_results = benchmark.generate_comprehensive_analysis()

recommendation_matrix = benchmark.create_recommendation_matrix()

print("\n✅ All tasks completed!")
