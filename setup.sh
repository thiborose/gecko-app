pip install -r requirements.txt

# Download gector model file
FILE=application/models/gector/data/model_files/xlnet_0_gector.th
if [ -f "$FILE" ]; then
    echo "$FILE is already downloaded."
else 
    (mkdir -p application/models/gector/data/model_files && cd application/models/gector/data/model_files && curl -O https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th)
fi


FILE=application/models/gector/data/model_files/xlnet_0_gector.th
if [ -f "$FILE" ]; then
    echo "$FILE is already downloaded."
else 
    (mkdir -p application/models/gector/data/model_files && cd application/models/gector/data/model_files && curl -O https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th)
fi


# Download sentence reordering models
FOLDER=application/models/sentence_reorder/model/
if [ -f "$FILE" ]; then
    echo "Sentence reordering models already downloaded."
else 
    (mkdir -p application/models/sentence_reorder/model/ && cd application/models/sentence_reorder/model/ && curl -O http://tts.speech.cs.cmu.edu/sentence_order/nips_bert.tar && unzip nips_bert.tar && rm nips_bert.tar)
fi