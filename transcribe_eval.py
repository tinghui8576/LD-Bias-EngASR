import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import tempfile
import soundfile as sf
from datasets import Audio, load_from_disk
from transformers import pipeline
import nemo.collections.asr as nemo_asr
from LD_utils.eval import metrics

def main(args):
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_id = args.model_name
    model_type = ""
    asr_pipe = None
    canary_model = None

    # --- 1. Unified Model Loading ---
    if "canary" in model_id.lower():
        print(f"Loading NVIDIA Canary model: {model_id}")
        canary_model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_name=model_id)
        canary_model.to(args.device)
        canary_model.eval()
        model_type = "canary"
    else:
        # Identify if it's Whisper or Wav2Vec for post-processing
        model_type = "whisper" if "whisper" in model_id.lower() else "wav2vec"
        print(f"Loading {model_type} via Transformers Pipeline...")
        
        # This pipeline handles chunking, striding, and feature extraction automatically
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if args.device.type == "cuda" else -1, # Pipeline expects int or -1
            chunk_length_s=25,  
            stride_length_s=(4, 2)
        )

    # --- 2. Dataset Processing ---
    for dset in args.eval_datasets:
        print(f'\nInferring on the dataset: {dset}')
        dataset = load_from_disk(dset)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
        
        filenames, all_preds, all_performances = [], [], []

        for item in tqdm(dataset, desc=f'Decoding ({model_type})'):
            audio_array = item["audio"]["array"]
            
            # --- 3. Inference Branching ---
            if model_type == "canary":
                # Canary requires a file path or manifest
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, audio_array, 16000)
                    # Canary output is a list of strings
                    result = canary_model.transcribe(
                        [tmp.name], 
                        source_lang=args.language, 
                        target_lang=args.language
                    )
                    pred = result[0] if isinstance(result[0], str) else result[0].text
            else:
                output = asr_pipe(audio_array.copy())
                pred = output["text"]

            # --- 4. Metrics & Storage ---
            performance = metrics(pred, item["sentence"])
            
            all_preds.append(pred)
            all_performances.append(performance)
            filenames.append(item["ID"])

        # --- 5. Save Results ---
        performance_df = pd.DataFrame.from_records(all_performances)
        base_df = pd.DataFrame({"id": filenames, "prediction": all_preds})
        outputs_df = pd.concat([base_df, performance_df], axis=1)

        output_csv_path = os.path.join(args.output_dir, f"{args.outputs}.csv")
        outputs_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="openai/whisper-small", 
                        help="Huggingface ID or NVIDIA Canary ID")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--eval_datasets", type=str, nargs='+', required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="predictions_dir")
    parser.add_argument("--outputs", type=str, default="results")

    args = parser.parse_args()
    main(args)