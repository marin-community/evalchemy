accelerate launch --num-processes 1 --num-machines 1 -m eval.eval \
    --model hf \
    --task alpaca_eval \
    --model_args 'pretrained=mistralai/Mistral-7B-Instruct-v0.3' \
    --batch_size 16 \
    --finestore_output_path logs/$(date -u +%Y%m%dT%H%M%S)-$$

