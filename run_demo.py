from social_monitor import SocialMediaMonitor
from multilingual_research import MultilingualSentimentResearch
from model_benchmark import ModelBenchmark

# Assignment 1
social_monitor = SocialMediaMonitor()
social_monitor.generate_sample_data(50)
analysis_results = social_monitor.run_complete_analysis()
social_monitor.create_visualizations()
social_monitor.generate_report()

# Assignment 2
research = MultilingualSentimentResearch()
experiment_results = research.conduct_practical_experiments()
performance_df = research.analyze_cross_lingual_performance(experiment_results)
recommendations = research.generate_recommendations()
paper_framework = research.compile_research_paper()

# Assignment 3
benchmark = ModelBenchmark()
benchmark_data = benchmark.generate_synthetic_benchmark_data()
benchmark.create_performance_visualizations()
analysis_results = benchmark.generate_comprehensive_analysis()
recommendation_matrix = benchmark.create_recommendation_matrix()
# run_demo.py - Add your code here
