pip install -r requirements.txt


FILE=application/models/gector/data/model_files/xlnet_0_gector.th
if [ -f "$FILE" ]; then
    echo "$FILE is alreday downloaded."
else 
    (mkdir -p application/models/gector/data/model_files && cd application/models/gector/data/model_files && curl -O https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th)
fi
