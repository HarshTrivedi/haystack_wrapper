{
    "query_model": "facebook/dpr-question_encoder-single-nq-base",
    "passage_model": "facebook/dpr-ctx_encoder-single-nq-base",
    "data_dir": "processed_data/sample",
    "train_filename": "dpr_training_data.json",
    "dev_filename": "dpr_training_data.json",
    "max_processes": 128,
    "batch_size": 4,
    "embed_title": true,
    "num_hard_negatives": 1,
    "num_positives": 1,
    "n_epochs": 10,
    "learning_rate": 1e-5,
    "num_warmup_steps": 10,
    "grad_acc_steps": 8,
    "optimizer_name": "AdamW",
    "evaluate_every": 100,
    "checkpoint_every": 100,
    "checkpoints_to_keep": 5,

    "index_type": "IVF_FLAT",
}