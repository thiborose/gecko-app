import argparse
from application.models.gector.gector.gec_model import GecBERTModel

def load_model(vocab_path,
               model_paths,
               max_len=50, 
               min_len=3,
               iterations=5,
               min_error_probability=0.0,
               min_probability=0.0,
               lowercase_tokens=0,
               model_name='bert',
               special_tokens_fix=0,
               log=False,
               confidence=0.0,
               is_ensemble=0,
               weights=None):

     model = GecBERTModel(vocab_path,
                         model_paths,
                         max_len=max_len, min_len=min_len,
                         iterations=iterations,
                         min_error_probability=min_error_probability,
                         min_probability=min_error_probability,
                         lowercase_tokens=lowercase_tokens,
                         model_name=model_name,
                         special_tokens_fix=special_tokens_fix,
                         log=False,
                         confidence=confidence,
                         is_ensemble=is_ensemble,
                         weights=weights)
     return model
