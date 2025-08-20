# from src.llm import llama_feedback
# from src.utils import reorder_all_evasive_embeddings, reorder_hidden_states
# from src.feature import feature_extraction
# from src.feature.models import tfidf_classifier
# from src.viz import plot_prompt_metrics_with_annotations
# from src.llm.embedding_extractor import EmbeddingExtractor
# from src.llm.embedding_extractor import uncompress_multiple_npz_to_npy
from src.hybrid.hybrid_classifier import HybridClassifier
# from src.feature.models import feature_classifier
# from src.feature.write_pkl_to_csv import get_pkl_sizes
# from src.feature.models import feature_classifier
# from src.llm.evaluate.get_zero_shot_metrics import process_predictions
# from src.llm.organize_evasive_texts import remove_empty_entries
# from src.llm.evaluate.gpt2_perplexity_calculator import calculate_evasive_perplexities
# from src.llm.llama_feedback import get_llama3_evasive_feedback
# from src.llm.fine_tuned_classifier import make_evasive_predictions
# from src.feature.error_features import extract_error_features
# from src.llm.evaluate.get_llm_metrics import get_zero_shot_evasive_metrics
# from src.feature.models.feature_classifier import get_evasive_pred
# from src.viz import generate_summary_table, plot_relative_differences, plot_prompt_metrics_with_annotations
# from src.llm.evaluate.mixed_effects_model import build_stats_model
# from src.viz import generate_perplexity_table
# from src.llm.evaluate import mixed_effects_model
# from src.llm.organize_evasive_texts import get_text_statistics

if __name__ == '__main__':
    # generate_perplexity_table()
    # mixed_effects_model.build_stats_model()
    # feature_extraction.main()
    # llama_feedback.get_llama3_texts()
    # fine_tuned_classifier.load_and_evaluate_model()
    # feature_classifier.run_model()
    # get_pkl_sizes()
    # feature_classifier.run_model()
    # feature_extraction.extract_features_from_evasive_texts()
    # feature_classifier.visualize_features()
    # remove_empty_entries()
    # get_llama3_evasive_feedback()
    # get_zero_shot_evasive_metrics()
    # get_evasive_pred()
    # tfidf_classifier.visualize_tfidf_features()
    # sum_tab = generate_summary_table()
    # plot_relative_differences(sum_tab)
    # plot_prompt_metrics_with_annotations()
    # tfidf_classifier.run_model()
    # tfidf_classifier.evaluate_evasive_texts()
    # tfidf_classifier.visualize_tfidf_features()
    # build_stats_model()
    # get_text_statistics()

    # extractor = EmbeddingExtractor()
    # extractor.load_and_prepare_data()
    # use_fine_tuned = False
    # extractor.extract_embeddings(use_fine_tuned=use_fine_tuned)

    # debug_none_ids_thorough()
    
    # subset_size = 100000
    # use_fine_tuned = False
    # # should try scaling the data too
    # # Mean pool and simple model
    # train_hybrid_model(subset_size=subset_size, use_simple_model=True, use_fine_tuned=use_fine_tuned, pooling="mean_pool")

    # # First token and simple model
    # train_hybrid_model(subset_size=subset_size, use_simple_model=True, use_fine_tuned=use_fine_tuned, pooling="first_token")

    # # Mean pool and advanced model
    # train_hybrid_model(subset_size=subset_size, use_simple_model=False, use_fine_tuned=use_fine_tuned, pooling="mean_pool")

    # # First token and advanced model
    # train_hybrid_model(subset_size=subset_size, use_simple_model=False, use_fine_tuned=use_fine_tuned, pooling="first_token")

    # train_hybrid_model()
    clf = HybridClassifier()      # use your usual constructor args
    clf.load_model()              # pulls weights from disk
    clf.visualize_feature_importance()            # creates figures/shap_hybrid.png