# DPR Training / Indexing / Prediction using Haystack

A sample sample experiment:

```bash
# Set "DOCKER_VOLUME_DIRECTORY" in .env
python train_dpr.py sample_config

python milvus_runner.py start
python milvus_runner.py status # not necessary

# might need 'unset LD_LIBRARY_PATH'
python index_dpr.py create sample_config processed_data/sample_dpr_prediction_data.json # keeping index and predict data the same.
python predict_dpr.py sample_config sample_config___sample_dpr_prediction_data processed_data/sample_dpr_prediction_data.jsonl

python milvus_runner.py stop # not necessary.
```
