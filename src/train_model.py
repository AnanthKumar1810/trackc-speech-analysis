"""Reproduce the grouped-CV-selected model and its frozen test evaluation."""
from experiment_classifier import main as compare_candidates
from retrain_model import main as fit_selected_model
from evaluate_retrained_model import main as evaluate_frozen_model

def main():
    compare_candidates()       # train/validation only; does not load test audio
    fit_selected_model()       # refit chosen model on train + validation
    evaluate_frozen_model()    # test split is loaded only after candidate freeze

if __name__=='__main__':
    main()
