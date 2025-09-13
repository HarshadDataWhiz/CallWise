from transformers import pipeline
import torch
import evaluate
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(file, model, processor, transcript_json, ROOT_FOLDER="."):
    """Worker function to process a single file."""
    try:
        # Initialize pipeline inside worker (important!)
        # transformers.pipeline objects are not pickleable → 
        # we need to initialize them inside each worker, not in the main process.
        vanilla_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=8,
            torch_dtype=torch.float32,
            device=-1  # force CPU since GPU is limited
        )

        wer_metric = evaluate.load("wer")

        audio_file_path = os.path.join(ROOT_FOLDER, file)

        # Run transcription
        prediction = vanilla_pipe(audio_file_path)
        predicted_text = prediction["text"]

        # Ground truth
        ground_truth = transcript_json[file]

        # Compute WER
        wer_score = wer_metric.compute(
            predictions=[predicted_text],
            references=[ground_truth]
        )

        return {
            "filename": file,
            "prediction": predicted_text,
            "ground_truth": ground_truth,
            "wer": wer_score,
            "error": None
        }

    except Exception as e:
        return {
            "filename": file,
            "prediction": None,
            "ground_truth": transcript_json.get(file, None),
            "wer": None,
            "error": str(e)
        }


def validation(model, processor, transcript_json, file_list, ROOT_FOLDER=".", max_workers=4):
    results = []

    # Run files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file, model, processor, transcript_json, ROOT_FOLDER): file
            for file in file_list
        }

        for future in as_completed(future_to_file):
            res = future.result()
            results.append(res)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results
