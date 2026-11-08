export WANDB_API_KEY=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
# bash finetune.sh place_can_basket_RDT
# bash finetune.sh place_bread_skillet_RDT
bash finetune.sh adjust_bottle_RDT
bash finetune.sh beat_block_hammer_RDT
bash finetune.sh blocks_ranking_rgb_RDT
bash finetune.sh blocks_ranking_size_RDT
bash finetune.sh click_alarmclock_RDT
bash finetune.sh click_bell_RDT
bash finetune.sh dump_bin_bigbin_RDT
bash finetune.sh grab_roller_RDT
bash finetune.sh handover_block_RDT
bash finetune.sh handover_mic_RDT
bash finetune.sh hanging_mug_RDT
bash finetune.sh lift_pot_RDT
bash finetune.sh move_can_pot_RDT
bash finetune.sh move_pillbottle_pad_RDT
bash finetune.sh move_playingcard_away_RDT
bash finetune.sh move_stapler_pad_RDT
bash finetune.sh open_laptop_RDT
bash finetune.sh open_microwave_RDT
bash finetune.sh pick_diverse_bottles_RDT
bash finetune.sh pick_dual_bottles_RDT

# bash eval.sh ${task_name} ${task_config} ${model_name} ${checkpoint_id} ${seed} ${gpu_id}
# bash eval.sh place_bread_skillet demo_aloha place_bread_skillet_RDT 10000 0 0
# bash eval.sh place_can_basket demo_aloha place_can_basket_RDT 10000 0 0

