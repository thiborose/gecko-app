FROM python:3.7

WORKDIR /app

# only copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /application/models/gector/data/model_files && cd application/models/gector/data/model_files && curl -O https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th
RUN mkdir -p /application/models/sentence_reorder cd /application/models/sentence_reorder && curl -O http://tts.speech.cs.cmu.edu/sentence_order/nips_bert.tar && tar -xf nips_bert.tar && rm nips_bert.tar && mv nips_bert/ model/

CMD ["python","run.py"]

EXPOSE 80
