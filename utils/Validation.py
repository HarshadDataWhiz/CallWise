from transformers import  pipeline
import torch
import evaluate
import os
import pandas as pd

def validation(model , processor, transcript_json, file_list, ROOT_FOLDER = "." ):
    vanilla_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=8,
        torch_dtype=torch.float32,
    )

    # Load WER metric
    wer_metric = evaluate.load("wer")

    results = []

    # Loop through all filenames
    for file in file_list:
        try:
            # File path
            audio_file_path = os.path.join(ROOT_FOLDER, file)

            # Transcription
            prediction = vanilla_pipe(audio_file_path)
            predicted_text = prediction["text"]

            # Ground truth
            ground_truth = transcript_json[file]

            # Compute WER
            wer_score = wer_metric.compute(
                predictions=[predicted_text],
                references=[ground_truth]
            )

            # Save result
            results.append({
                "filename": file,
                "prediction": predicted_text,
                "ground_truth": ground_truth,
                "wer": wer_score,
                "error": None
            })

        except Exception as e:
            print(f"Error processing {file}: {e}")
            results.append({
                "filename": file,
                "prediction": None,
                "ground_truth": transcript_json.get(file, None),
                "wer": None,
                "error": str(e)
            })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    return df_results

