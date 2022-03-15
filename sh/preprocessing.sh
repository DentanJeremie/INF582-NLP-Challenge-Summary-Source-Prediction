echo "Basic preprocessing of the raw data... (should save test_set.json and train_set.json in processed_data)"
python scripts/preprocessing.py
echo "Basic preprocessing : done !"
echo ""
echo ""
echo "Generating classification score with roberta model...  (should save roberta_train.csv and roberta_test.csv in processed_data)"
python scripts/features_generation/roberta-transformer/main_train_roberta.py
echo "Classification with roberta model : done !"
echo ""
echo ""
echo "Generating gltr features... (should save gltr_train.csv and gltr_test.csv in processed_data)"
python scripts/features_generation/gltr_method/gltr_features.py 0
echo "Generating gltr features : done !"
echo ""
echo ""
echo "Generating keywords features... (should save keywords_train.csv and keywords_test.csv in in processed_data)"
python scripts/features_generation/keywords_features.py
echo "Generating keywords features : done !"
echo ""
echo ""
echo "Generating embeding features... (should save embedding_train.csv and embedding_test.csv in in processed_data)"
python scripts/features_generation/embedding_features.py
echo "Generating embedding features : done !"
echo ""
echo ""
echo "Generating ngrams features... (should save ngrams_train.csv and ngrams_test.csv in in processed_data)"
python scripts/features_generation/ngrams_features.py
echo "Generating ngrams features : done !"
echo ""
echo ""
echo "Generating rouge features... (should save rouge_train.csv and rouge_test.csv in in processed_data)"
python scripts/features_generation/rouge_features.py
echo "Generating rouge features : done !"
echo ""
echo "PREPROCESSING : DONE !"

