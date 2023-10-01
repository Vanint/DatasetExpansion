export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="./organsmnist_finetuned"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --seed=2023 \
  --resolution=512 \
  --train_batch_size=2 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=0 \
  --sample_batch_size=4 \
  --max_train_steps=400 \
  --save_interval=200 \
  --save_sample_prompt="medical photo of heart" \
  --concepts_list="organsmnist_concepts_list.json"


#--with_prior_preservation --prior_loss_weight=1.0 \
